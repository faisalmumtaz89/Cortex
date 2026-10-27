---
name: review-changes
description: >-
  Use when asked only to review changes (a diff, branch, commit, pull request
  or uncommitted work) for bugs, regressions or security flaws. Proves every
  finding with a reproduction it runs, drops what it cannot prove, and ranks
  findings by severity. Reports only: edits are refused for the rest of the
  turn, so for "review and fix" it reports and fixes come next message.
allowed-tools: [read_file, list_dir, search, git, bash, write_file]
---
# Review changes

You are the adversary of this diff. Find what is really wrong with it, prove
each problem by running it, and report. Do not fix anything, and do not report
as proven what you could not prove.

## Process

1. **Get the diff** with the git tool, which needs no approval. Use what the
   user named. Otherwise use the uncommitted changes: `status`, and `diff`
   with revision `HEAD`, and read every untracked file it lists in full. If
   there are none, use the branch against its base: `merge-base` with revision
   `main` (or `master`, or `origin/main`) gives the base, then `diff` with
   revision `<base>...HEAD`. Start with `stat` to see the scope. If the diff
   is empty, say so and stop.
2. **Learn the intent** from the commit messages, any description the user
   gave, and the tests that changed.
3. **Read each changed hunk in context** with read_file, and search for the
   callers of every function whose behaviour or signature changed.
4. **Hunt.**
   - Logic: wrong conditions, off-by-one errors, inverted checks, wrong
     variables.
   - Inputs: empty, None, zero, negative, huge, unicode, duplicates.
   - Errors: swallowed exceptions, leaked resources, partial writes.
   - Callers: changed defaults, return values or raised errors that existing
     callers do not expect.
   - Concurrency and ordering.
   - Security: injection (SQL, shell, template), path traversal, missing
     authorization, secrets in code, unsafe deserialization, request forgery,
     weak crypto.
5. **Prove each candidate.** Write the smallest script that shows the wrong
   result, crash or exploit, and run it against the changed code. Put scratch
   files in `mktemp -d /tmp/cortex-review.XXXXXX` and reuse the literal path it
   prints: shell variables do not survive between bash calls. Prefer
   standalone scripts run as `PYTHONPATH=. python /tmp/cortex-review.…/repro.py`
   (or the project's own interpreter and import path); a test file outside the
   repository does not get the project's conftest.py fixtures. To compare with
   the old behaviour, get the old file with the git tool: `show` with revision
   `<base>:<path>`. For a
   secret in code, the proof is the search hit: its `path:line` and the kind of
   secret, with the value masked. A finding is confirmed only when the run
   shows the failure.
6. **Drop** style, naming and nitpicks, and problems that existed before the
   diff (mention a severe one in a single line at most).
7. **Report** the findings ranked by severity:
   - **Critical**: security hole, data loss.
   - **High**: wrong results or a crash in normal use.
   - **Medium**: wrong results in edge cases.
   - **Low**: minor but real.

   For each finding give a title, `path:line`, what goes wrong, the
   reproduction (the command and its output, trimmed), and the direction of a
   fix in one line. Then list briefly what you checked that held up. With no
   findings, say so and say what you checked. `examples/report.md` shows the
   format.

## Rules

- Never change the repository, except to write the review itself to a file
  when the user asks for that (use write_file, and only for that file). Read
  git state with the git tool; through bash run only read-only git commands
  it lacks (`blame`, `rev-parse`, `ls-files`, `grep`), and never write or
  delete repository files through bash either: no
  `sed -i`, redirects into the repository, formatters, package installs, or
  scripts that write there. If the user also asked for fixes, report, and say
  the fixes come in the next message.
- Running the project's code or tests can write files (caches, snapshots,
  `.coverage`): check that the git tool's `status` matches what it was at the
  start.
- If bash is refused or unavailable, you cannot prove anything. Say so first,
  list each candidate under **Unverified** with its `path:line` and the command
  that would prove it, and never write that no problems were found.
- Remove the temporary directory by its literal `/tmp/cortex-review.…` path.
- A failing reproduction you ran is the only proof. Reasoning alone is not.
