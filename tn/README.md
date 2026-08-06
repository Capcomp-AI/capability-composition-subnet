# Testnet recipes

Recipes committed on the testnet arena (netuid 544), published here so the
pointer in each on-chain commitment resolves. A miner hosts these wherever they
like; this directory is only where these particular ones live.

They are built against the miniature pool, not the certified one, because the
testnet engine runs the protocol on a CPU with a simulated model. The merge, the
digest and the commitment are real; the scores they earn measure the harness.

A commitment payload is capped at 128 bytes, of which 57 are the magic, workflow
code and digest. That leaves **71 characters for the URI** — short enough that
`raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>` usually will not fit.
`https://github.com/<owner>/<repo>/raw/<branch>/<path>` is 11 characters shorter
and redirects to the same bytes.
