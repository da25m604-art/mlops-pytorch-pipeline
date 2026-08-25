# Reflection write-up

## What was the most challenging part?

The hardest part of this assignment was not writing the model or the manifests —
those followed fairly directly from the starter code and the Kubernetes docs.
The real difficulty was everything *around* the code: the workflow, the
environment split, and the feedback loops that only surface once things are
actually running.

The single biggest challenge was working across two machines. I did all the Git
and authoring on my Windows laptop, but Docker and Kubernetes realistically had
to run on a Linux GCP VM. This meant that every feature branch had to be pushed
from Windows and then pulled onto the VM before I could build or deploy — and
more than once I tried to `docker build` on the VM only to find the `docker/`
folder wasn't there yet because it was still an untracked file on Windows. It
forced me to be disciplined about the branch-per-slice workflow: commit a slice,
push it, pull it on the VM, verify, then open the PR with the evidence.

Training time was the next surprise. ResNet-18 on CPU took roughly 18 minutes
per epoch, which is fine locally but painful inside a Kubernetes Job that has to
run to completion before serving can start. I learned to separate "does the
mechanism work" from "is the model good," and used a smaller CNN with fewer
epochs for the cluster demo while keeping the ResNet config as the default.
Related to this, the training Job appeared to hang for ten minutes in the
cluster — it turned out CIFAR-10 was re-downloading onto the data PVC over a
slow pod network, and the progress bar simply wasn't flushing to `kubectl logs`.
Learning to diagnose that with `kubectl describe pod` and `kubectl exec ... ls`
instead of assuming the pod was broken was a genuinely useful lesson.

Finally, CI was quietly failing on every run. The `lint-and-test` job was dying
in a few seconds because `actions/setup-python` couldn't find a root
`requirements.txt` for its pip cache, and once that was fixed, ruff flagged a
couple of small lint issues that had been masked. Debugging GitHub Actions from
the logs, rather than guessing, was the fix.

If I did it again I would set up the cluster and CI first, before writing much
code, so the slow feedback loops were in place from the start. The main takeaway
is that in MLOps the model is the easy part — the packaging, orchestration, and
automation around it are where most of the real effort and most of the
debugging live.
