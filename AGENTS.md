# novakit agent instructions

## Project

`novakit` is a standalone, embeddable Python Agent SDK. It provides LLM providers,
streaming events, tool execution, sessions, approvals, and an in-memory agent
runtime. It must remain usable without Nova or any Nova installation.

## Commands

```bash
uv sync --extra dev
uv run pytest -q --cov=novakit --cov-report=term-missing
uv run ruff check novakit tests
uv run ruff format novakit tests
uv run mypy
uv build --wheel --out-dir dist
bash scripts/smoke_wheel.sh
```

Use `uv` for Python dependency management. Run the relevant tests, Ruff, and mypy
before declaring work complete.

## Hard boundaries

- Never import `nova` or any `nova.*` module from `novakit`.
- Do not read `HOME`, `NOVA_HOME`, or `~/.nova` by default.
- Do not create user directories or connect to a database by default.
- Default persistence is in memory. Hosts inject persistence through
  `SessionProtocol`.
- Keep one implementation of the agent turn loop, `ToolInvoker`, and shared
  session service. Do not copy them into adapters.
- Every announced tool call must end with a result, error, rejection, or
  cancellation result.

## Compatibility surface

Nova is the first host consumer. Treat these as compatibility surfaces even when
they are protected or protocol-based:

- `Agent(config, llm, *, options=None)`
- `Agent._initialize_runtime(...)`
- `AgentRuntimeDependencies`
- `AgentOptions` and `AgentPorts`
- `CoreConfig` and `AgentConfig`
- `SessionProtocol`
- `ApprovalManager` and `Compactor`
- `chat_stream` event order and payload shapes

Prefer additive, backward-compatible changes. Breaking changes require a major
version bump and migration notes.

## Code quality

- Python 3.12+; use modern type annotations.
- All public functions, classes, and methods require docstrings and type hints.
- Avoid `Any` unless the data is genuinely unstructured.
- Use context managers for resources and specific exception types.
- Keep functions focused and avoid mutable default arguments.
- Use Ruff for formatting and linting; use mypy for static typing.
- Tests use pytest and the Arrange-Act-Assert pattern.
- Mock external services; default tests must be offline and fast.

## Git

- Do not commit or push unless the user explicitly asks.
- Use English Conventional Commits with no `Co-authored-by` trailers.
- Never rewrite history or discard user work without confirmation.

## Definition of done

- Relevant tests pass.
- Full test suite passes with at least 99% line coverage for `novakit`.
- Ruff and mypy pass.
- Wheel build and isolated install smoke pass.
- No Nova imports or user-directory defaults were introduced.
- Tool terminal pairing remains intact.
- README, changelog, and handoff notes are updated when public behavior changes.
