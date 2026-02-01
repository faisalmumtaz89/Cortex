# Conversation Management

## Overview

Cortex stores conversation history in memory and (by default) persists it to SQLite for recovery and export. The implementation lives in `cortex/conversation_manager.py`.

## Storage and Autosave

When `conversation.auto_save` is `true`, Cortex writes conversations to:

```
~/.cortex/conversations/conversations.db
```

The CLI `/save` command exports the current conversation as JSON to:

```
~/.cortex/conversations/conversation_<timestamp>.json
```

## Message Model

Conversations are composed of `Message` objects:

- `role`: `system`, `user`, or `assistant`
- `content`: raw text
- `timestamp`: ISO timestamp
- `message_id`: unique identifier

## Context Handling

`Conversation.get_context(max_tokens=...)` can trim history to a token budget, but the current CLI uses **full conversation history** when formatting prompts. There is no automatic sliding‑window truncation in the CLI today.

## Branching (Data Model)

The `Conversation.branch(...)` method can create a new conversation from an earlier message, but branching is **not exposed in the CLI** yet.

## Export Formats

The conversation manager supports:

- `json` (used by `/save`)
- `markdown` (available via API, not exposed in CLI)
