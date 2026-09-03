# docker

One file, kept for one reason.

`Dockerfile.runner` builds the sandbox runner image, and the engine's
`evaluator_image_digest` is the digest of what it produces. Every published
evaluation report carries that digest, so a third party replaying a run needs
this file to know what the digest attests to. Deleting it would leave a hash in
every report with nothing on the other end.

Nothing here is required to run a miner or a validator, and nothing in the test
suite needs a Docker daemon. The engine and console deployments build from their
own repository.
