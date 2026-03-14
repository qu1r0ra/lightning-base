# Notes

## Todos

- Invite project collaborators
- Import the [protected-main ruleset](../protected-main.json) to GitHub `Settings -> Rules -> Rulesets`
- Uncomment commented [.gitignore](../../.gitignore) entries under _Miscellaneous_
- Install project dependencies with `uv sync` (see <https://docs.astral.sh/uv/reference/cli/#uv-sync>)
- Set up Git pre-commit hooks with `pre-commit install` (see <https://pre-commit.com/>)
- > will add more
- Finish setting up the rest of the codebase

## Persistent Checklist

- **Documentation & Context Files** (Requires constant updates to reflect project state):
  - `README.md` (Project overview, installation, and user guide)
  - `AGENTS.md` (Root entry point and high-level directives for AI assistants)
  - `docs/ARCHITECTURE.md` (Technical design overview and component interactions)
  - `docs/agents/CONTEXT.md` (Definitive guide to tech stack, coding standards, and constraints)
  - `docs/agents/GEMINI.md` (Gemini-specific operational instructions)
  - `docs/agents/CLAUDE.md` (Claude-specific operational instructions)
  - `misc/context/METACONTEXT.md` (Persistent memory for discoveries, problems, and architectural reasoning)
  - `Makefile` (Primary task runner and repository automation suite)
  - `tests/` (Hierarchical test suite for unit and integration verification)
  - `src/lightning_uv_wandb_template/utils/constants.py` (Centralized configuration for paths, defaults, and hyperparameters)

## Prompts

### New Chat (Before)

i will create a new chat since this chat has become too heavy for my device.
before i do so, make sure all files in the persistent checklist in `misc/personal/NOTES.md`
are updated, comprehensively reflecting the project's state.

### New Chat (After)

go through the entire codebase to deeply understand it. there are various markdown files like

`README.md`
`AGENTS.md`
`docs/ARCHITECTURE.md`
`docs/agents/CONTEXT.md`
`docs/agents/GEMINI.md`
`docs/agents/CLAUDE.md`
`misc/context/METACONTEXT.md`

which may help. ask any questions you have.

## Commands

Counting `data/` class instances

```bash
for d in */; do echo -n "$d: "; find "$d" -maxdepth 1 -type f | wc -l; done
```
