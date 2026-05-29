# 4DSTEM Processing Library Project Rules

This document is the handoff anchor for the 4DLAB 4DSTEM processing library.
Before adding or changing code, read these rules first.

## Core Rules

1. Record every code change in detail.
   - After writing or modifying code, update `docs/WORK_LOG.md`.
   - The log should include date, changed files, purpose, implementation notes, verification, and next steps.
   - The goal is that a new conversation can understand the project state quickly and continue work safely.

2. Keep feature areas isolated.
   - Different functions should live in separate modules, packages, or folders.
   - Adding or modifying one feature should avoid disturbing unrelated features.
   - Shared utilities are allowed only when they are truly general and stable.

3. Keep code organized by responsibility.
   - Source code, tests, examples, documentation, scripts, and data templates should each live in their matching folders.
   - Do not scatter files in random locations.
   - New folders should have clear names that describe their responsibility.

4. Keep the project portable.
   - The library should run on another computer after installing the documented environment.
   - Avoid hard-coded local paths, machine-specific settings, or hidden dependencies.
   - Dependencies, setup steps, and required runtime versions must be documented.

## Suggested Folder Layout

```text
4DLAB/
  docs/             Project rules, work logs, design notes, user notes
  src/              Library source code
  tests/            Automated tests
  examples/         Small runnable examples and notebooks
  scripts/          Utility scripts for setup, conversion, or batch work
  configs/          Portable configuration templates
  data/             Small sample data or metadata only; avoid large raw datasets
```

## Change Log Checklist

When code changes are made, append an entry to `docs/WORK_LOG.md` with:

- Date and time
- Author or assistant/session
- Goal
- Files changed
- Main design choices
- How it was verified
- Known limitations
- Suggested next step

