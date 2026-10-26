# Conversation Management

## Overview

Cortex keeps each session's conversation and, by default, saves it to `~/.cortex/conversations/conversations.db`; `/save` also writes it as a JSON file, and `/clear` starts over. In the interactive app a session's conversation continues across turns; each headless `-p` run starts a new one. (For contributors: the implementation is in `cortex/conversation_manager.py`, and `cortex/app/session_service.py` maps sessions to conversations.)

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
- `parts`: for assistant replies, the text, reasoning, tool calls, and tool results in the order they happened

## Context Handling

Every turn sends the whole conversation to the model, after the system prompt, including the tool calls each reply made, their results, and the reasoning the model streamed for each step (saved in the message's `parts`). An interrupted reply keeps the tool calls that finished before the interrupt; a turn that fails with an error is not saved, except a reply cut off at the output limit, at `tools_max_iterations`, or at the context window, which keeps what it wrote. Tool calls and results from the current turn and the two before it are sent in full; in older turns each result and each long tool argument keeps only its first and last 750 characters. Reasoning is always sent in full, as DeepSeek requires. When a conversation outgrows the model's context window, its turns fail; `/clear` starts a fresh one.

## Export

`/save` writes the conversation as JSON. There is no other export format, and no branching of a conversation, yet.
