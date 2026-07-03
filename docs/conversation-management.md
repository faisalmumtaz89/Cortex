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

## Context Handling

Agent turns send a sliding window of the most recent messages (the last 30 non-empty messages) to the model, on top of the system prompt. `Conversation.get_context(max_tokens=...)` can also trim history to a token budget via the API.

## Branching (Data Model)

`Conversation.branch(...)` can create a new conversation from an earlier message, but branching is **not exposed in the UI** yet.

## Export Formats

- `json` (used by `/save`)
- `markdown` (available via `export_conversation(..., format="markdown")`, not exposed in the UI)
