# AGENTS

## Standard instructions for coding and related tasks:

- Prefer simplicity and clarity in code and explanations, while still being thorough and complete.
- Always write type lists for all functions and similar constructs but not simple variables.
- Always write Google-style docstrings for functions and other code elements.
- Prefer clear and explicit code with good variable names and comments over terseness or cleverness.
- Prefer modular code with well-defined functions and classes over monolithic scripts.
- Prefer using standard libraries and well-known third-party libraries over custom implementations, unless there is a clear reason to do otherwise.
- Always consider the context of the project (the scientific idea/domain) and the specific requirements of the task.
- Always consider the maintainability and readability of the code for future users and developers, including yourself.
- Consider the trade-offs between different approaches and explain the reasoning behind your suggestions.
- Feel free to ask clarifying questions if the task or requirements are not clear, or if you need more information to provide a good answer.
- Feel free to create new files or directories if needed for your own experimentation to help me in any way possible, but do not modify or delete any existing files or directories (those created by me) without explicit asking me to do so.
- Always explain in simple words the logic and reasoning of the code just implemented, and how it relates to the scientific idea/domain and the specific requirements of the task. The changes or improvements you made, etc. I should know what you did and why, and how it helps the project. This is important for me to understand the code and to learn from it, and also for future reference when I or others look at the code later on.
- Always consider the best practices and conventions of the programming language and the scientific domain, and try to follow them as much as possible, unless there is a good reason to deviate from them.
- Always use tab indentation for code blocks, and avoid mixing tabs and spaces.

## Repository Lookup Policy

- When suggesting commands, workflows, scripts, or implementation patterns, check known repos first.
- For ALCHEMY tasks: check `ALCHEMY__dev` first, then `ALCHEMY__main`.
- For planet evolution tasks: check `PLANET_EVO__main`.
- For generic utility/shell/workflow tasks: check `HELPFUL_SCRIPTS` first, then `https://github.com/akashgpt/helpful_scripts`.
- There is no single global order across unrelated project domains.
- If project-local code conflicts with external repos, prioritize this project's local code/config.
- If a referenced repo/path is unavailable in the current session, state that explicitly and continue with best available sources.

## Local Skill Routing

- Treat the repo-local `SKILL.md` files under `HELPFUL_SCRIPTS/qmd` as local onboarding guides that should be loaded into context for `qmd` work.
- For any task inside `HELPFUL_SCRIPTS/qmd`, first read `qmd/SKILL.md`.
- Then read the most relevant folder-specific skill file before doing substantive work:
  - `qmd/vasp/SKILL.md`
  - `qmd/TI/SKILL.md`
  - `qmd/ALCHEMY/SKILL.md`
  - `qmd/plmd/SKILL.md`
  - `qmd/setup_INPUT/SKILL.md`
- If the task spans more than one `qmd` subfolder, read all relevant skill files and keep their workflow guidance in mind during the task.
- When a folder skill points to a `references/` file, read that only when the task depends on external software behavior, file-format semantics, or official syntax.
- For ALCHEMY tasks, still follow the repository lookup policy above and check `ALCHEMY__dev` first; the `qmd/ALCHEMY` skill is a helper-layer guide, not the canonical pipeline definition.

## Local Priority Repositories

| Env var            | Local path                                                | Remote                                            | Notes                                 |
| ------------------ | --------------------------------------------------------- | ------------------------------------------------- | ------------------------------------- |
| `ALCHEMY__dev`     | `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev`     | `git@github.com:akashgpt/DPAL.git`                | ALCHEMY active development copy       |
| `ALCHEMY__main`    | `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__in_use`  | `git@github.com:akashgpt/DPAL.git`                | ALCHEMY in-use/stable copy            |
| `PLANET_EVO__main` | `/projects/BURROWS/akashgpt/run_scripts/planet_evo_x_qmd` | `git@github.com:akashgpt/planet_evo_x_qmd.git`    | planet_evo private repo (local clone) |
| `HELPFUL_SCRIPTS`  | `/projects/BURROWS/akashgpt/run_scripts/helpful_scripts`  | `https://github.com/akashgpt/helpful_scripts.git` | Utility scripts repo                  |

## Benchmarks Reference

- For benchmark-related questions, templates, performance comparisons, or cluster-specific run settings, check `$HELP_SCRIPTS/benchmarks` first.
- The benchmarks folder contains curated benchmark notes, input templates, submission scripts, compilation files, and timing/output summaries for:
  - VASP: `benchmarks/vasp/<cluster>/` including NCSA Delta, Stellar, Della, and Tiger references.
  - DeePMD: `benchmarks/deepmd/<cluster>/` including GPU benchmark scripts and representative inputs.
  - LAMMPS/PLUMED: `benchmarks/lammps/<cluster>/` where available.
- For NCSA Delta VASP work, prefer `$HELP_SCRIPTS/benchmarks/vasp/NCSA_DELTA` for known-good compilation notes, submission templates, GPU/CPU scaling observations, and convergence artifacts before creating new scripts.
- Treat benchmark folders as reference material: copy/adapt scripts into working directories as needed, but avoid overwriting benchmark records unless the user explicitly asks to update them.

## Cross-cluster SSH Access (Princeton: stellar / tiger / della)

**Hosts & aliases.** `tiger` and `della` are defined as SSH aliases in `~/.ssh/config`. From stellar, commands on the other two clusters run as `ssh tiger '<cmd>'` / `ssh della '<cmd>'`.

**Path invariant (relied on across clusters).**
- User scratch root: `/scratch/gpfs/BURROWS/akashgpt/` — same absolute path on stellar, tiger, and della.
- User home: `/home/ag5805/` — same on all three.
- Paths under scratch/home can therefore be constructed identically regardless of which cluster is the target.

**Data is NOT mirrored.** Identical top-level folder names across clusters do NOT imply identical contents or layout. Always diff before assuming overlap.

**Auth model & the main failure mode.**
- Princeton enforces DUO 2FA on first connect. A fresh `ssh <host> '<cmd>'` from a script/agent will fail with `Permission denied (keyboard-interactive)` unless a prior interactive session has been authenticated (another terminal) or a ControlMaster socket is active.
- Recovery: ask the user to run `ssh tiger` / `ssh della` once in another terminal; subsequent non-interactive `ssh <host> '<cmd>'` calls then succeed.
- If a tiger (or della) hostname changes (e.g. after a cluster rebuild), expect a `REMOTE HOST IDENTIFICATION HAS CHANGED` warning. Do NOT edit `~/.ssh/known_hosts` silently — surface it and let the user decide.

**Recommended `~/.ssh/config` stanza for reusable non-interactive access:**
```
Host <alias>
    User ag5805
    HostName <fqdn>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```
After one manual `ssh <alias>` to authenticate, subsequent commands reuse the socket for 8h without re-prompting.

**Inventory / overlap-diff pattern.**
```bash
# Dump a listing from each cluster, including remote ones via ssh.
# %y = file type (d/f/l), %p = path. Use -maxdepth N to control depth.
DUMP="find <path> -maxdepth 2 -printf '%y %p\n' | LC_ALL=C sort"
eval "$DUMP"                    > /tmp/inv_stellar.txt
ssh tiger "$DUMP"               > /tmp/inv_tiger.txt
ssh della "$DUMP"               > /tmp/inv_della.txt

# Compare (inputs must be LC_ALL=C-sorted).
comm -12 /tmp/inv_stellar.txt /tmp/inv_tiger.txt   # shared entries
comm -23 /tmp/inv_stellar.txt /tmp/inv_tiger.txt   # only on stellar
comm -13 /tmp/inv_stellar.txt /tmp/inv_tiger.txt   # only on tiger
```

**Quick reachability check (use before driving remote work):**
```bash
ssh -o ConnectTimeout=10 <host> 'hostname; id -un' 2>&1 | head -5
```
If this hangs or returns the permission-denied message, the session isn't warm — stop and ask the user to authenticate.

## GitHub Repository Index (`akashgpt`)

Public repositories for `https://github.com/akashgpt` (snapshot generated on `2026-02-22T00:48:31Z` via GitHub API).

Source endpoint:

- `https://api.github.com/users/akashgpt/repos?per_page=100&type=owner&sort=updated`

| Repo            | URL                                         | Type     | Primary language | Last updated (UTC)   |
| --------------- | ------------------------------------------- | -------- | ---------------- | -------------------- |
| helpful_scripts | https://github.com/akashgpt/helpful_scripts | Original | Python           | 2026-02-13T21:32:40Z |
| mldp            | https://github.com/akashgpt/mldp            | Fork     | Python           | 2024-11-15T23:43:57Z |
| eos             | https://github.com/akashgpt/eos             | Fork     | Unspecified      | 2024-10-29T18:21:19Z |
| chems           | https://github.com/akashgpt/chems           | Fork     | Unspecified      | 2023-10-06T01:30:55Z |

## Notes

- This index covers public repos visible via the GitHub API for user `akashgpt`.
