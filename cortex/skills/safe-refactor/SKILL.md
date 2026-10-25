---
name: safe-refactor
description: >-
  Use to restructure code without changing what it does (rename, extract,
  move, deduplicate, simplify) or to upgrade third-party dependencies. Starts
  only from a passing test suite, changes in small steps, re-runs the tests
  after each, and keeps public APIs and behaviour the same unless the user asks
  otherwise. Not for fixing failing tests or bugs.
---
# Safe refactor

Green to green. The suite passes before you start, after every step, and at
the end. What the code does for its callers stays the same.

**The gate:** run the whole suite before any edit. If a single test fails or
errors, anywhere, even in code you will not touch, you do not refactor in this
turn: your answer is the list of failures and the advice to fix them first.

## Process

1. **Baseline.** Find the project's test command (AGENTS.md, README, CI
   config, `pyproject.toml`, `package.json`, `Makefile`) and run the full suite.
   Give long suites a longer bash timeout (up to 600000 ms); if a run still
   times out, run it one directory at a time. Run it without `-x`, so you see
   every failure. If any test fails, apply the gate above: report the failures
   and stop. Fixing them is a separate task. Otherwise note the number of
   passing tests.
2. **Write down the contract**, everything that must not change: public
   names and signatures (exports, `__all__`, documented APIs), import paths
   other code uses, CLI flags, output and file formats, error types, and the
   inputs the code accepts today, odd ones included. Search the whole
   repository for each name you will touch, including tests, docs, configs,
   entry points and strings.
3. **Cover what you will change.** If no test exercises the code you will
   move or rewrite, first add characterization tests that pin its current
   behaviour, odd behaviour included, and run them green.
4. **Plan small steps** that each leave the code working. For example:
   extract the new function, switch callers one module at a time, then remove
   the old code.
5. **Take one step at a time.** Make the change, then run the tests that cover
   it, and the full suite at each milestone. If a step goes red, fix it, or
   undo your own edit with edit_file. Never use `git checkout`, `restore`,
   `stash` or `reset`: they also discard the user's uncommitted work. Never
   stack changes on a red suite.
6. **Keep public names.** When a public name moves or is renamed, keep the old
   one as an alias or re-export unless the user asked to break it. Update
   every reference you found in step 2.
7. **Upgrade one dependency at a time.** Read its changelog or release notes
   for every version between the old and the new one; if you cannot reach
   them, say so and ask before upgrading. List the breaking changes that touch
   APIs this code uses (search for them). Change the pin with the project's
   own package manager so the lockfile is updated, install only into the
   project's virtual environment (stop and ask if there is none), run the
   suite, fix the uses, then go on to the next dependency.
8. **Finish.** Run the full suite: at least as many tests pass as at the
   baseline. Review `git diff`: no behaviour change, no unrelated edits, no
   weakened test. Report each step, the test results before and after, and
   anything you deliberately left alone.

## Rules

- Never change a test's expected values to make it pass. A test may change
  only its import of a moved name.
- No new features or bug fixes mixed in. A bug you find is reported for
  later, not fixed here.
- Do not reformat code you are not otherwise changing.
