# Conversation Management

## Overview

Cortex stores conversation history in memory and (by default) persists it to SQLite for recovery and export. The implementation lives in `cortex/conversation_manager.py`; the worker maps each session to a conversation in `cortex/app/session_service.py`.

## Storage and Autosave

When `auto_save` is `true` (the default), conversations are written to:

```
~/.cortex/conversations/conversations.db
```

The `/save` command exports the current conversation as JSON to:

```
~/.cortex/conversations/conversation_<timestamp>.json
```

`/clear` starts a fresh conversation for the session.

## Message Model

Conversations are composed of `Message` objects:

- `role`: `system`, `user`, or `assistant`
- `content`: raw text
- `timestamp`: ISO timestamp
- `message_id`: unique identifier
- `parts`: for assistant replies, the text, tool calls, and tool results in the order they happened

## Context Handling

Every turn sends the whole conversation to the model, after the system prompt, including the tool calls each reply made and their results (saved in the message's `parts`). An interrupted reply keeps the tool calls that finished before the interrupt; a turn that fails with an error is not saved, except a reply cut off at the output limit or the context window, which keeps the text it wrote. Tool calls and results from the current turn and the two before it are sent in full; in older turns each result and each long tool argument keeps only its first and last 750 characters. When a conversation outgrows the model's context window, its turns fail; `/clear` starts a fresh one. `Conversation.get_context(max_tokens=...)` can also trim history to a token budget via the API.

## Branching (Data Model)

`Conversation.branch(...)` can create a new conversation from an earlier message, but branching is **not exposed in the UI** yet.

## Export Formats

- `json` (used by `/save`)
- `markdown` (available via `export_conversation(..., format="markdown")`, not exposed in the UI)
