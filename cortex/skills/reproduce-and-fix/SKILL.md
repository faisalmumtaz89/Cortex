---
name: reproduce-and-fix
description: >-
  Use when the user wants something broken fixed: a bug report, a failing
  test, an error or traceback, or a failing CI log. Finds the first real error,
  reproduces it with a failing test, fixes the code rather than the test, and
  re-runs the full suite. Not for only explaining an error, new features,
  refactors, or updating tests the user says are outdated.
---
# Reproduce and fix

Red, then green. Do not edit source code until a test fails for the reported
reason. Then fix the code, never the test, and prove it with the full suite.

## Process

1. **Find the first real error.** In a log, skip setup output. The first
   failure is usually the cause; later failures often follow from it. Note the
   failing test or command, the exception and its message, and the innermost
   stack frame in the project's own code. Warnings are noise unless they are
   the failure. `examples/ci-log.md` shows how to read a CI log.
2. **Find how the project runs its tests.** Check AGENTS.md, README,
   CONTRIBUTING, the CI config (`.github/workflows/`), and `pyproject.toml`,
   `package.json` or `Makefile`. Use the same command CI uses.
3. **Take a baseline.** Before any edit, run the full suite once and note the
   tests that fail. Those failures already existed. Give long suites a longer
   bash timeout (up to 600000 ms); if a run still times out, run it one
   directory at a time.
4. **Reproduce it.**
   - A failing test was given: run exactly that test. If it passes here, do
     not edit anything: compare the CI environment (versions, environment
     variables, test order) with yours, and report what differs.
   - Only a report was given: write the smallest test that shows the bug, where
     and how the project writes its tests, and run it.
   - A lint, type-check or build error: the failing command is the
     reproduction.

   The test must fail **for the reported reason**. Read the failure: an import
   error, a typo or a missing fixture in your test is not a reproduction.
   Record the command and the failure message.
5. **Rule out a flake.** If the failure involves time, randomness, ordering,
   network or concurrency, run it three times. If it passes sometimes, it is a
   flake: report it, do not fix it here, and go back to step 1 for the next
   real error.
6. **Find the cause.** Read the code on the failing path. Form one hypothesis
   and confirm it before editing: read the code, narrow the test, or run a
   scratch script. For a regression, `git log -p -- <file>` and
   `git show <commit>` find the change that broke it. Do not check out other
   commits or run `git bisect`, which move HEAD; if bisecting is really
   needed, ask the user to run it.
7. **Fix the implementation** at the cause, not where the symptom shows, with
   the smallest change that fixes it. Never edit the failing test to pass,
   weaken an assertion, skip it, mark it xfail, or catch and ignore the error.
   If the test itself is wrong (it contradicts documented behaviour), stop and
   say so instead of changing it.
8. **Verify.** Run the reproduction, which must pass now. Then run the full
   suite. If another test fails, compare it with the baseline from step 3.
   Never use `git stash`, `checkout` or `reset` to compare; they can hide or
   lose uncommitted work. Failures that were in the baseline are reported, not
   fixed.
9. **Report** the cause (`path:line`), the fix, the test that proves it, and
   each command you ran with its result. Keep the new test. Delete scratch
   files.

## Rules

- No source edit before a failing reproduction. If the bug cannot be
  reproduced here (it needs a service, data or a platform you lack), say so,
  say what you tried, and stop without editing source.
- Keep the diff to the bug: no refactors or style changes along the way.
- Never claim a test passed unless you ran it and saw it pass.
