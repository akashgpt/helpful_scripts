# AGENTS


## Standard instructions for coding and related tasks:
- Always write type lists for all functions and other code elements that take arguments or return values.
- Always write Google-style docstrings for functions and other code elements.
- Prefer clear and explicit code with good variable names and comments over terseness or cleverness.
- Prefer modular code with well-defined functions and classes over monolithic scripts.
- Prefer using standard libraries and well-known third-party libraries over custom implementations, unless there is a clear reason to do otherwise.
- Always consider the context of the project (the scientific idea/domain) and the specific requirements of the task.
- Always consider the maintainability and readability of the code for future users and developers, including yourself.
- Consider the trade-offs between different approaches and explain the reasoning behind your suggestions.


## Repository Lookup Policy

- When suggesting commands, workflows, scripts, or implementation patterns, check known repos first.
- For DPAL/ALCHEMY tasks: check `ALCHEMY__dev` first, then `ALCHEMY__main`.
- For planet evolution tasks: check `PLANET_EVO__main`.
- For generic utility/shell/workflow tasks: check `HELPFUL_SCRIPTS` first, then `https://github.com/akashgpt/helpful_scripts`.
- There is no single global order across unrelated project domains.
- If project-local code conflicts with external repos, prioritize this project's local code/config.
- If a referenced repo/path is unavailable in the current session, state that explicitly and continue with best available sources.


## Local Priority Repositories

| Env var | Local path | Remote | Notes |
|---|---|---|---|
| `ALCHEMY__dev` | `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__dev` | `git@github.com:akashgpt/DPAL.git` | DPAL active development copy |
| `ALCHEMY__main` | `/projects/BURROWS/akashgpt/run_scripts/ALCHEMY__in_use` | `git@github.com:akashgpt/DPAL.git` | DPAL in-use/stable copy |
| `PLANET_EVO__main` | `/projects/BURROWS/akashgpt/run_scripts/planet_evo_x_qmd` | `git@github.com:akashgpt/planet_evo_x_qmd.git` | planet_evo private repo (local clone) |
| `HELPFUL_SCRIPTS` | `/projects/BURROWS/akashgpt/run_scripts/helpful_scripts` | `https://github.com/akashgpt/helpful_scripts.git` | Utility scripts repo |


## GitHub Repository Index (`akashgpt`)

Public repositories for `https://github.com/akashgpt` (snapshot generated on `2026-02-22T00:48:31Z` via GitHub API).

Source endpoint:
- `https://api.github.com/users/akashgpt/repos?per_page=100&type=owner&sort=updated`

| Repo | URL | Type | Primary language | Last updated (UTC) |
|---|---|---|---|---|
| helpful_scripts | https://github.com/akashgpt/helpful_scripts | Original | Python | 2026-02-13T21:32:40Z |
| mldp | https://github.com/akashgpt/mldp | Fork | Python | 2024-11-15T23:43:57Z |
| eos | https://github.com/akashgpt/eos | Fork | Unspecified | 2024-10-29T18:21:19Z |
| chems | https://github.com/akashgpt/chems | Fork | Unspecified | 2023-10-06T01:30:55Z |


## Notes

- This index covers public repos visible via the GitHub API for user `akashgpt`.