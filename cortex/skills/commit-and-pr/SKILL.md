---
name: commit-and-pr
description: >-
  Use when asked to commit, push, or open or describe a pull request, once the
  code changes are done. Checks the diff for leaked secrets and debug
  leftovers, writes clear commit messages and PR descriptions, and runs git
  safely: no force pushes, no history rewrites. It never changes source or
  test files; if the request also asks for code changes, make them first.
allowed-tools: [read_file, list_dir, search, git, bash, write_file]
---
# Commit and PR

Package finished work: check it, commit it, and push or describe it only when
asked. You never change source or test files. If they need a change, stop and
say what. The one file you may write is one the user asks for to hold a commit
message or pull request description (such as `PR_BODY.md`): use write_file
for it, and only for it.

## Process

1. **Look at the state** with the git tool, which needs no approval: `status`
   (it names the branch), `log`, `diff`, and `diff` with `staged`. Run
   `git remote -v` with bash only when you will push. Decide what belongs in
   the commit: by default, the changes of the task at hand. Leave unrelated
   changes uncommitted and say which you left. If you are on `main` or `master` and
   the user did not ask to commit there, create a branch with
   `git switch -c <short-name>` and say so.
2. **Scan the lines being added** (the git tool's `diff`, staged and not, the
   `+` lines, and every new file in full).
   - **Blockers**: do not commit; stop and report. These are secrets (API keys
     and tokens such as `sk-…`, `ghp_…`, `AKIA…`, `xox…`; `-----BEGIN … PRIVATE
     KEY-----`; passwords or tokens assigned real-looking values), credential
     files and `.env` files (not `.env.example` or `.env.sample`), debugger
     calls (`breakpoint()`, `pdb.set_trace()`, `debugger;`), focused tests
     (`.only(`, `fit(`, `fdescribe(`), and merge conflict markers. Report the
     file and line and the kind of secret. Never repeat a secret's value.
   - **Warnings**: commit, but list them in your reply. These are new prints or
     console logs that look like debugging, and TODOs that were added.
   - Files that should not be committed at all (scratch files, build output,
     generated files, local config) are left unstaged; suggest a `.gitignore`
     entry.
3. **Stage by path**: `git add -- <path> …`. Use `git add -A` or `git add .`
   only when every change was reviewed and belongs. If the index already holds
   changes you did not stage, stop and ask; do not unstage them. Check with
   the git tool: `diff` with `staged` and `stat`.
4. **Write the message** in the repository's style. Check the git tool's
   `log` for its prefixes, case and length. Otherwise:
   - a subject of at most 72 characters, in the imperative ("Fix …", not
     "Fixed …"), with no trailing period;
   - a blank line;
   - a body wrapped at 72 characters that says what was wrong or missing, what
     changed, and what follows from it, not a list of files.

   Commit with `git commit -F - <<'EOF'`, the message, then `EOF`. Quote
   `'EOF'` so backticks and `$` stay literal. `examples/message.md` shows a
   good message.
5. **Commit and verify** with the git tool: `log` with `stat` and `max_count`
   1, and `status`.
6. **Push only when asked**: `git push -u origin <branch>`. If the push is
   rejected (the remote has commits you do not have), stop and report. Do not
   rebase, merge or force.
7. **Pull request, only when asked.** Asking for a pull request is also asking
   to push: push as in step 6 first. Look for a template
   (`.github/pull_request_template.md`, `.github/PULL_REQUEST_TEMPLATE/`,
   `docs/pull_request_template.md`) and fill in its sections. Otherwise write:
   - Summary: what and why;
   - Changes: short bullets;
   - Testing: only the commands that were really run, with their results;
   - Risks or notes.

   If `gh` is installed and logged in, run `gh pr create --base <base> --head
   <branch> --title "…" --body-file - <<'EOF'`, the description, then `EOF`.
   Otherwise give the title and description for the user to paste.

## Never

- Force pushes in any form (`--force`, `--force-with-lease`, `-f`, `--mirror`,
  a refspec starting with `+`), or deleting remote refs (`--delete`,
  `:branch`).
- `reset`, `commit --amend`, `rebase`, `filter-branch`; `restore`,
  `checkout -- <path>`, `clean` or `stash`, which discard or hide the user's
  work; deleting branches or tags; changing git config.
- Interactive commands: `-i`, `-p`, or `git commit` without `-F`, which opens
  an editor and hangs.
- `--no-verify`. If a hook fails, report its output; if it changed files, list
  them (the git tool's `status`), do not stage them, and stop.
- Committing to `main` or `master`, or pushing to them, unless the user asked
  for exactly that.
- Editing source or test files, even to remove a debug line, and writing or
  deleting repository files through bash (`sed -i`, redirects, formatters).
  Report what needs to change instead. write_file is only for a message or
  description file the user asked for.
