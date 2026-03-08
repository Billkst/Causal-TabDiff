---
description: Before using `npx skills`, verify `npx` is on PATH in the current shell; if not, inspect conda environments/package cache and only proceed after activating or exporting a working Node/npm runtime with required shared libraries.
triggers:
  - user asks to use find-skills or run `npx skills`
  - shell reports `npx: command not found`
  - task depends on Node/npm tooling inside conda-managed environments
confidence: true
---

# Instinct: Resolve `npx` via Conda Before Skill Search

## Mistake Captured
Assuming `npx` is globally available in the default shell caused a false conclusion that the skill lookup itself failed. In this workspace, `npx` was not on the default PATH, even though Node/npm artifacts existed inside the conda installation.

## Root Cause Pattern
1. The current shell was only using the default environment and did **not** expose `npx` on PATH.
2. The relevant Node runtime was present only in conda package locations, not an activated env bin directory.
3. Even direct execution of cached `npx` failed until Node's dependent shared libraries (such as `libuv` and ICU libs) were added through `LD_LIBRARY_PATH`.
4. Therefore, `npx: command not found` did **not** mean the requested skill was absent; it meant the execution environment was incomplete.

## BAD Pattern
```bash
npx skills find skill-lookup
# bash: npx: command not found
```

## GOOD Pattern
```bash
# 1) Check first
command -v npx || true
command -v conda || true

# 2) If missing, locate a working conda-provided Node/npm runtime
find /home/UserData/miniconda -path '*/bin/node' -o -path '*/bin/npm' -o -path '*/bin/npx'

# 3) Export both executable and library paths before retrying
export PATH=/home/UserData/miniconda/pkgs/nodejs-20.17.0-hb8e3597_0/bin:$PATH
export LD_LIBRARY_PATH=/home/UserData/miniconda/pkgs/nodejs-20.17.0-hb8e3597_0/lib:/home/UserData/miniconda/pkgs/libuv-1.48.0-h5eee18b_0/lib:/home/UserData/miniconda/pkgs/icu-73.2-h59595ed_0/lib:${LD_LIBRARY_PATH:-}

# 4) Validate, then run the skills query
node --version
npx --version
npx skills find skill-lookup
```

## Action Rule
1. **Never** interpret `npx: command not found` as proof that a skill does not exist.
2. **Always** validate `npx` availability before any `find-skills` workflow.
3. If `npx` is missing, **always** check conda environments and the conda package cache for Node/npm binaries.
4. If Node is launched from a package cache path, **always** verify dependent shared libraries with `ldd` and export the needed library directories before retrying.
5. Only after `node --version` and `npx --version` succeed may the skill search result be trusted.

## Completion Signal
Report all of:
1. whether `npx` was originally on PATH,
2. what runtime path was used,
3. whether extra library paths were required,
4. the final `npx skills` result.
