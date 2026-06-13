# CLAUDE.md

Behavioral guidelines and project conventions for Croscim.

**Tradeoff:** these guidelines bias toward caution over speed. For trivial
tasks, use judgment.

For full project context, commands, caveats, and review priorities, read
[AGENTS.md](AGENTS.md).

## 1. Think Before Coding

**Do not assume. Do not hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them. Do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what is confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: would a senior engineer say this is overcomplicated? If yes,
simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Do not improve adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it. Do not delete it.

When your changes create orphans:

- Remove imports, variables, or functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

## 5. Project Conventions

**Language and style**

- Chat with the user in French unless asked otherwise.
- Code and comments should follow the existing file language and style.
- Keep prose direct. If a sentence does not add information, delete it.

**Data safety**

- Never commit data files, model checkpoints, generated figures, Hydra outputs,
  TensorBoard logs, or large run logs.
- Do not delete raw notes in `notes/`.
- Treat machine data roots such as `/nwp/sst_malegu`,
  `/dmidata/projects/4dvarnet`, and `/dcai/projects/cu_0026` as operational
  paths. Do not run destructive operations there unless explicitly requested.

**What not to do**

- Do not silently change scientific assumptions, channel order, normalization,
  patch geometry, or loss definitions.
- Do not touch application code during documentation-only tasks.
- Do not use raw notes as source of truth without checking current code.
- Do not commit data files or model checkpoints to git.

## 6. Croscim Reminder

Croscim trains a multi-resolution 4D-VarNet model for global SST reconstruction
from daily satellite Zarr data.

Maintained documentation:

- [README.md](README.md)
- [AGENTS.md](AGENTS.md)
- [docs/](docs/)

Raw notes in `notes/` and historical files in `archive/docs/` are context, not
the current source of truth.
