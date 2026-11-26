# Sessions

`InMemorySession` (in `novakit/session/memory.py`) is the default persistence.
Each instance owns its sessions, messages, and current-session `ContextVar`.
No files are created and no database is opened.

## Default usage

```python
from novakit import Agent, AgentConfig
from novakit.llm import FakerLLMProvider
from novakit.session import InMemorySession

session = InMemorySession()
agent = Agent(
    AgentConfig(model="fake", max_iterations=4),
    FakerLLMProvider(seed=1),
)
```

Pass the session through `AgentOptions(session=...)` when you want to own the
instance explicitly; otherwise the `Agent` creates one for you. Either way the
data stays in process memory.

## Injecting your own persistence

Implement `SessionProtocol` (`novakit/session/protocol.py`) and inject it:

```python
from novakit import AgentOptions

options = AgentOptions(session=my_session)
agent = Agent(config, llm, options=options)
```

`SessionProtocol` is the seam the turn loop programs against: session lookup
and creation, message persistence, title handling, and route metadata. A
database host provides a thin adapter; the core never assumes SQL, files, or a
user directory.

## Message and session models

`novakit/session/models.py` defines the plain data types flowing through the
seam:

- `SessionContext` — session row (id, title, workspace, route, fork lineage,
  turn count); `SessionContext.create(...)` builds a new one.
- `NewMessage` — a message to persist (role, content, tool calls, token
  counts, provider metadata, timestamps).
- `Message` — a persisted message with its assigned id.
- `Session` — a lightweight session view.
- `MessageFilter` — query filter for listing messages.
- `encode_display_text` / `decode_display_text` — display-text helpers.

Example: building a message to persist via a fork store (same `NewMessage`
shape the protocol uses):

```python
from novakit.session.models import NewMessage

pending = NewMessage(session_id=session_id, role="user", content="Hello")
```

Hosts that need branching should use `fork_session` with a `SessionForkStore`
instead of copying rows by hand — see [Session fork](session-fork.md).

## API reference

::: novakit.session.SessionProtocol
::: novakit.session.InMemorySession
::: novakit.session.Message
::: novakit.session.NewMessage
::: novakit.session.SessionContext
