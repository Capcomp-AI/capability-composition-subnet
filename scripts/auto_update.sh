#!/usr/bin/env bash
# Keep a node on the current protocol revision, and put it back if that fails.
#
# Every participant runs consensus-relevant code. A validator on an old spec
# derives a different window than the network decides; a miner on an old
# contract writes recipes that admission rejects. So falling behind is not a
# cosmetic problem, and neither is updating badly.
#
# What this does NOT do is trust the remote blindly:
#
#   * fast-forward only. A rewritten remote history stops the update rather than
#     rewriting local state, because the two are indistinguishable from here and
#     one of them is an attack.
#   * the remote URL is checked before fetching. A repointed origin is the
#     cheapest way to feed a node someone else's code.
#   * every incoming commit has to carry a trusted author *and* committer, and
#     is checked before the merge, so untrusted code never becomes HEAD.
#   * the new revision has to import and pass the fast suite *before* anything is
#     restarted. A node that updates into a broken tree and restarts is worse
#     than one that never updated.
#   * if the units do not come back, the previous revision is restored and
#     restarted. Ending in a state nobody chose is the failure mode worth
#     engineering against.
#
# Worth being plain about the limit. Author and committer are strings the pusher
# chooses, and git does not authenticate them, so the identity check below stops
# the wrong identity pushing — not somebody who can push and can also type a
# different name. Until these identities sign their commits, what is really
# verified is the remote, the fast-forward, and that the code works here.
# CAPSUB_REQUIRE_SIGNED=1 closes that gap the day signing starts.
set -uo pipefail

REPO="${CAPSUB_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BRANCH="${CAPSUB_UPDATE_BRANCH:-main}"
EXPECTED_REMOTE="${CAPSUB_UPDATE_REMOTE:-https://github.com/Capcomp-AI/capability-composition-subnet.git}"
UNITS="${CAPSUB_UPDATE_UNITS:-}"          # space-separated; empty is valid (a miner has none)
PIP="${CAPSUB_PIP:-pip}"
EXTRAS="${CAPSUB_INSTALL_EXTRAS:-[merge]}"
SMOKE="${CAPSUB_UPDATE_SMOKE:-1}"         # 0 disables the pre-restart test run

# Identities whose commits this node will run. Space-separated emails; both the
# author and the committer of every incoming commit must appear here.
TRUSTED_AUTHORS="${CAPSUB_TRUSTED_AUTHORS:-josiah.dev521@gmail.com xinyangtaylor@gmail.com}"

# Require a good signature on every incoming commit as well. Off by default
# because this repository's history is unsigned and turning it on would refuse
# every update; turn it on the day these identities start signing, and identity
# stops being a claim and becomes a key.
REQUIRE_SIGNED="${CAPSUB_REQUIRE_SIGNED:-0}"
UNIT_SETTLE_SECONDS="${CAPSUB_UNIT_SETTLE_SECONDS:-45}"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR $*"; exit 1; }

cd "$REPO" || die "no repository at $REPO"
git rev-parse --git-dir >/dev/null 2>&1 || die "$REPO is not a git repository"

actual_remote="$(git remote get-url origin 2>/dev/null || true)"
if [ "$actual_remote" != "$EXPECTED_REMOTE" ]; then
  die "origin is $actual_remote, expected $EXPECTED_REMOTE. Refusing to update from an unexpected remote."
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  die "the working tree has local changes. Refusing to update over them."
fi

before="$(git rev-parse HEAD)"

git fetch --quiet origin "$BRANCH" || die "could not fetch origin/$BRANCH"
after="$(git rev-parse "origin/$BRANCH")"

if [ "$before" = "$after" ]; then
  log "already on ${after:0:12}, nothing to do"
  exit 0
fi

# Who wrote it, checked before anything is merged — the objects are already
# fetched, so this rejects untrusted code without ever moving HEAD onto it.
#
# Both author and committer, because they differ: a commit written by a trusted
# identity and rebased, amended or cherry-picked by somebody else carries the
# first name and the second one's content.
#
# Be clear about what this is worth. Author and committer are strings the pusher
# chooses; git does not authenticate them. This stops a commit from an identity
# that should not be pushing here — a stray account, a machine with the wrong
# gitconfig, a contributor pushing to the wrong remote — and it does not stop
# anyone who can push and can also type a different name. The control that does
# is a signature: set CAPSUB_REQUIRE_SIGNED=1 once these identities sign, and
# this becomes a check on a key rather than on a claim.
untrusted=""
while IFS='|' read -r sha aemail cemail sig; do
  [ -z "$sha" ] && continue
  # Author and committer are usually the same person; report the pair once so a
  # single bad commit reads as one problem rather than two.
  identities="$aemail"
  [ "$cemail" != "$aemail" ] && identities="$aemail $cemail"
  for who in $identities; do
    case " $TRUSTED_AUTHORS " in
      *" $who "*) ;;
      *) untrusted="$untrusted ${sha:0:12}($who)" ;;
    esac
  done
  if [ "$REQUIRE_SIGNED" = "1" ] && [ "$sig" != "G" ] && [ "$sig" != "U" ]; then
    untrusted="$untrusted ${sha:0:12}(unsigned:$sig)"
  fi
done <<EOF
$(git log --format='%H|%ae|%ce|%G?' "$before..origin/$BRANCH")
EOF

if [ -n "$untrusted" ]; then
  log "ERROR refusing revisions not from a trusted identity:$untrusted"
  log "      trusted: $TRUSTED_AUTHORS"
  exit 1
fi

# Fast-forward only. If the remote rewrote history this fails, and it should:
# from here a force-push and a compromise look identical.
if ! git merge --ff-only --quiet "origin/$BRANCH"; then
  die "origin/$BRANCH is not a fast-forward of ${before:0:12}. History diverged or was rewritten; not updating."
fi

log "updating ${before:0:12} -> ${after:0:12}"
git --no-pager log --format='  %h %s' "$before..$after" | head -20

spec_before="$(git show "$before:capability_subnet/__init__.py" 2>/dev/null | grep -oE '__version__ = "[^"]+"' || true)"
spec_after="$(grep -oE '__version__ = "[^"]+"' capability_subnet/__init__.py || true)"
if [ -n "$spec_before" ] && [ "$spec_before" != "$spec_after" ]; then
  log "NOTE version changed: $spec_before -> $spec_after (consensus-relevant; other nodes must follow)"
fi

roll_back() {
  log "rolling back to ${before:0:12}"
  git reset --hard --quiet "$before" || log "ERROR could not roll back; this node needs a human"
  $PIP install -q -e ".${EXTRAS}" >/dev/null 2>&1 || log "ERROR reinstall after rollback failed"
  [ -n "${running:-}" ] && systemctl restart $running 2>/dev/null
  return 0
}

if ! $PIP install -q -e ".${EXTRAS}"; then
  log "ERROR install failed on ${after:0:12}"
  roll_back
  exit 1
fi

if [ "$SMOKE" = "1" ]; then
  # Cheap, and it is the difference between updating and updating into a tree
  # that cannot score anybody.
  if ! python -c "import capability_subnet, capability_subnet.validator.evaluator" 2>/dev/null; then
    log "ERROR ${after:0:12} does not import"
    roll_back
    exit 1
  fi
  if ! make test-fast >/tmp/capsub-update-smoke.log 2>&1; then
    log "ERROR fast suite failed on ${after:0:12}; see /tmp/capsub-update-smoke.log"
    tail -20 /tmp/capsub-update-smoke.log | sed 's/^/    /'
    roll_back
    exit 1
  fi
  log "smoke check passed"
fi

if [ -z "$UNITS" ]; then
  # A miner submits and exits, so there is nothing to restart — the next run
  # picks this up. Saying so beats silence, which reads as a broken updater.
  log "no units configured; the next run will use ${after:0:12}"
  exit 0
fi

# Only what is already running. A unit an operator stopped — because the host has
# no GPU yet, because they are mid-maintenance — must not be started by an
# updater, and must not be counted as a failed restart afterwards. Without this,
# a node with one deliberately-stopped unit rolls back every update it is offered
# and never moves, reporting a rollback rather than the reason.
running=""
for unit in $UNITS; do
  if [ "$(systemctl is-active "$unit" 2>/dev/null || true)" = "active" ]; then
    running="$running $unit"
  else
    log "leaving $unit alone; it was not running before this update"
  fi
done
running="${running# }"

if [ -z "$running" ]; then
  log "no configured unit was running; updated to ${after:0:12} without restarting anything"
  exit 0
fi

log "restarting:$running"
systemctl restart $running || { log "ERROR restart failed"; roll_back; exit 1; }

sleep "$UNIT_SETTLE_SECONDS"

failed=""
for unit in $running; do
  state="$(systemctl is-active "$unit" 2>/dev/null || true)"
  [ "$state" = "active" ] || failed="$failed $unit($state)"
done

if [ -n "$failed" ]; then
  log "ERROR did not come back:$failed"
  roll_back
  exit 1
fi

log "updated to ${after:0:12};$running active"
