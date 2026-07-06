# Slurm pending-reason & failed-state reference

## Pending reasons (`squeue` / `scontrol show job` → `Reason=(...)`)

| Reason code | Meaning | Smallest safe fix |
|---|---|---|
| `(Resources)` | Waiting for free nodes/CPUs/GPUs — normal | Wait, or request fewer resources / a less busy partition |
| `(Priority)` | Lower priority than other queued jobs | Wait; check your fairshare (`sshare -U`) |
| `(QOSMaxCpuPerUserLimit)` | You hit your per-user QOS CPU cap | Let running jobs finish, or request fewer CPUs |
| `(MaxSubmitJobsPerAccount)` | The shared account hit its job-count cap | Too many jobs queued on the account; wait or coordinate with labmates |
| `(AssocGrpCpuLimit)` | The account/association CPU limit is reached | Reduce CPUs, or wait for the group's jobs to drain |
| `(ReqNodeNotAvail,UnavailableNodes:...)` | Pinned to a node that is down or drained | Drop the `-w`/`--nodelist` constraint so any node can run it |
| `(PartitionTimeLimit)` | `--time` exceeds the partition's max | Lower `--time`, or move to a partition with a longer limit |

## Failed / ended states (`sacct -j <id> --format=State,ExitCode,Elapsed,MaxRSS`)

| State | Likely cause | What to check |
|---|---|---|
| `OUT_OF_MEMORY` / `OOM` | Job exceeded `--mem` | Raise `--mem`; inspect `MaxRSS` from `sacct` |
| `TIMEOUT` | Ran past `--time` | Increase `--time` (before submit) or checkpoint the work |
| `NODE_FAIL` | Hardware/node problem, not your fault | Resubmit; report the node if it recurs |
| `FAILED` (ExitCode ≠ 0:0) | Your program errored | Read the `.err` log; reproduce interactively |
