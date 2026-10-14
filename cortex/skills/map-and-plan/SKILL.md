---
name: map-and-plan
description: >-
  Use when the user asks how code works, where something happens, or wants a
  change planned but not yet made: maps entry points, data flow and key
  modules with exact path:line references, then writes a step-by-step edit
  plan. Read-only for the whole turn: not for requests that also ask to make
  the change, run commands, or fix something now.
allowed-tools: [read_file, list_dir, search]
---
# Map and plan

Answer with a map of the code the request touches and, when a change is asked
for, a plan to make it. In this turn you only read: edits and commands are
refused. The plan is the deliverable; the user runs it in a later message.

## Process

1. **Pin down the request.** Restate it in one sentence. If it can be read two
   ways that lead to different code, say which reading you map.
2. **Find the entry points.** Search for what a user sees first: command and
   flag names, routes, public functions, config keys, error messages. List the
   package roots to see the layout. Read the README, AGENTS.md or docs/ only as
   far as they explain the architecture you need.
3. **Trace the flow.** From each entry point, follow the calls to where the
   behaviour happens. Search for definitions and callers instead of guessing
   them. Read the functions you will cite, and only the part of a large file
   you need. Stop following a branch once it leaves the area the request
   touches, and stop exploring once you can answer.
4. **For a change, find what constrains it:** the tests that cover this code
   (search tests/ for the module and function names), the closest existing
   feature built the same way (the pattern to copy), and every caller that a
   changed signature or behaviour would break.
5. **Check every reference.** Each `path:line` you write must come from a
   read_file or search result in this turn and point at the line that defines
   or does the thing. Drop any reference you did not see.
6. **Write the answer** in the format below.

## Answer format

**Summary**: two to four sentences that answer the question.

**Map**: the flow in order, one line per step:
`path:line`: `symbol`, then what happens there.

**Plan** (only when a change is asked for): numbered steps in the order they
must happen. Each step names the file and function (with `path:line`), says
what to change, and says why. Include the tests to add or update and the
command that runs them, cited from where you found it (CI config,
`pyproject.toml`, `Makefile`). End with the risks: callers that must change
too, behaviour that others depend on, data or config that must migrate.

**Open questions**: only what the code cannot answer and the user must decide.

`examples/plan.md` shows a complete answer; if you are unsure of the level of
detail, read it with the skill tool.

## Rules

- Never guess. Write an unknown as unknown, and say what you searched for.
- Code outside the repository, such as installed packages, cannot be read:
  mark what depends on it as unknown.
- A few exact references beat many vague ones. Cite lines; do not paste long
  code.
- Do not claim you changed, ran or tested anything. You did not.
