# Repositories and what belongs in them

Miners and validators clone this repository and run from it. The obvious next
question is what the subnet owner keeps that they do not — and the answer is
narrower than it first looks, because most of what an operator runs has to be
public for the subnet's central guarantee to hold.

## One public repository

This one. It contains the protocol contracts, the merge engine, the workflow
generators and scorers, the sandbox, **the evaluation engine**, and the three
neurons.

The engine being public is not an oversight or a transparency gesture. It is
load-bearing. The subnet's answer to "why should anyone believe a single
operator's scores" is that every score is published with the trace it came from,
every closed window discloses its seeds, and instance generation is a pure
function of the seed — so anyone can regenerate the exact problems a candidate
faced and re-run the scoring. Validators do this on every pass before paying.

That check only works if the scorer is the same scorer. Move the scoring code
into a private repository and "re-score the window" becomes "re-score it with
whatever scorer you happen to have", which is not a check at all. **A private
engine would not be a more secure subnet; it would be an unverifiable one.**

The same argument covers the workflow generators. They look like the secret and
they are not: the secret is `hidden_seed_root`, a single integer. Publishing the
generator while keeping the seed lets an auditor reproduce a closed window and
still leaves future windows unpredictable. Publishing the seed would publish
every future hidden instance, which is why it is the one value the engine
refuses to start without having been changed.

## One private operations repository

What an operator genuinely holds back is not code. It is:

| Kept private | Why |
|---|---|
| `hidden_seed_root` | Publishing it publishes every future hidden instance. |
| Wallet files, coldkey/hotkey handling | Ordinary key hygiene. |
| `backend.yaml` and `.env` as *filled in* | They carry the seed root, database credentials, and host detail. |
| GPU host inventory, SSH config, provisioning | Infrastructure, not protocol. |
| Monitoring dashboards, alert routing, on-call runbooks | Operational, and useless to anyone else. |
| Incident notes | Frequently contain the above. |

None of it is a fork of this repository. The right shape is a small private repo
that *depends* on the public one:

```
capability-subnet-ops/          (private)
├── inventory/                  which host serves which GPU
├── config/backend.yaml         filled in, secrets by reference
├── deploy/                     compose overrides, systemd units
├── monitoring/                 dashboards and alerts
└── runbooks/                   what to do when the engine stalls
```

It pins a released version of the public package and configures it. Nothing in
it changes how a candidate is scored — which is the test for whether something
belongs there at all. **If a change to the private repo could change a miner's
score, it is in the wrong repo.**

## The one case that genuinely needs a private workflow

A future workflow built on a real customer's business process cannot have its
generator published, because the generator *is* the process. That is a real
requirement and this repository now supports it without a fork: workflows are
discovered through the `capability_subnet.workflows` entry point group, so a
private distribution can register one and the engine will load it.

```toml
# in the private workflow distribution's pyproject.toml
[project.entry-points."capability_subnet.workflows"]
acme_claims_intake_v1 = "acme_workflow:load"
```

Use it knowing exactly what it costs. A workflow nobody outside the operator can
install is a workflow nobody outside the operator can replay, and replay is the
entire basis on which a validator pays a centralised engine. Such a workflow
therefore declares `publicly_verifiable=False`, the engine logs it loudly, and
the network is told plainly that scores on it rest on the operator's word.

For V1 the workflow is synthetic and public, and it should stay that way for as
long as possible.

## Splitting *this* repository

Not yet, and the reason is specific. The merge engine, the workflow generators
and the scorers must be version-locked together or consensus breaks: an auditor
replaying a closed window with a differently-versioned scorer produces a
different answer, and the disagreement looks like operator fraud rather than a
packaging mistake. Today `spec_version` binds them and one test suite crosses
every boundary.

The dependency seam that *does* matter is already enforced —
`tests/unit/test_layering.py` holds `common`, `workflows`, `miner`, `validator`,
`audit` and `platform` free of the tensor stack, which is what lets a validator
install in around 50 MB. That is the practical benefit a split would have
delivered, obtained without paying for it in version drift.

Revisit when a second workflow ships. At that point workflows become
independently versioned arenas with their own release cadence, the entry point
mechanism above is already the interface, and extracting a
`capability-subnet-protocol` distribution starts paying for itself.
