# Session fork

`novakit/session/fork.py` copies a session and its whole-message prefix into a
new session. The source is never mutated. Hosts implement the narrow
`SessionForkStore` protocol; `novakit` owns the copy algorithm.

## Forking a session

```python
from novakit.session.fork import fork_session

result = await fork_session(store, session_id, message_id=None)
print(result["session_id"], result["messages"])
```

Arguments:

- `store` — host persistence implementing `SessionForkStore` (`get_session`,
  `get_messages`, `save_session`, `set_session_route`, `add_message`).
- `session_id` — source session to copy.
- `message_id` — last message to include; omitted copies all visible history.
- `title` — title for the copy, or `"<origin title> (fork)"` when omitted
  (suffix constant `FORK_SUFFIX = " (fork)"`).
- `default_agent_key` — fallback owner when the source row has no agent key
  (defaults to `novakit.constants.DEFAULT_AGENT_KEY`).

The result describes the copy: `session_id`, `title`, message count,
`forked_from`, and `fork_point_message_id`.

## Copy semantics

- **Whole-message prefix** — `_messages_up_to` returns messages through
  `message_id` inclusive; a fork never slices mid-history. An unknown
  `message_id` raises `ForkPointNotFound`; a missing source raises
  `FileNotFoundError`.
- **Tool pairing preserved** — each message is re-added with its full
  `tool_calls` / `tool_call_id`, token counts, provider metadata, reasoning
  fields, and timestamps, so tool-call/result pairs stay intact.
- **Metadata and route copied** — the fork inherits workspace, project,
  provider, model, and reasoning effort via `set_session_route`, records
  `forked_from` / `fork_point_message_id`, and sets `turn_count` from the
  copied user messages.

## API reference

::: novakit.session.fork.fork_session
::: novakit.session.fork.SessionForkStore
::: novakit.session.fork.ForkPointNotFound
