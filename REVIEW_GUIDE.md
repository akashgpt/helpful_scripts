# Auto-Review Boundaries

When I ask for auto-review, follow these boundaries unless I explicitly override them.

## Allowed Without Asking

- Reading files under `$HELP_SCRIPTS`.
- Reading files under `$ALCHEMY__main`.
- Reading files under `$SCRATCH`.
- Running non-destructive inspection commands such as: `rg`, `find`, `sed`, `awk`, `git status`, `git diff`, `git log`, `bash -n`.
- Checking job status.
- Submitting new jobs.
- Canceling jobs you started.
- Keep ONGOING folder up to date (only for the cluster/system you are working on)
- Create new files
- Edit any files you created.
- Never exceed 100 GB of total created output without explicit permission.
- Always, summarize findings in chat.

## Ask First

- Editing any existing file.
- Deleting files.
- Running `sbatch`, `srun`, or long background jobs.
- Running `git pull`, `git push`, `git fetch`, `git commit`.
- Installing packages or using network access.

## Never Do During Auto-Review

- Edit files created by others.
- Edit files that are not within the folders: "/projects/BURROWS/akashgpt", "/scratch/gpfs/BURROWS/akashgpt", "/projects/bguf/akashgpt/", or "/work/nvme/bguf/akashgpt".
- Delete or overwrite files under `/scratch`.
- Submit or cancel Slurm jobs.
- Change Git remotes or credentials.

## VASP/MLFF Review Preferences

- Treat `INCAR` as the primary settings file.
- Check `RUN_VASP.sh` for cluster/runtime consistency.
- Check `POSCAR` species order against `POTCAR` only by metadata unless I ask for deeper inspection.
- Prefer sampling logs over scanning full `OUTCAR`.
- Report exact file paths and line numbers when possible.
