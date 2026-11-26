# Installation

`novakit` requires **Python 3.12 or newer** (`requires-python = ">=3.12"` in
`pyproject.toml`, tested on 3.12 and 3.13).

## From PyPI

```bash
pip install novakit
```

## With uv

```bash
uv add novakit
```

## Optional extras

Defined in `pyproject.toml` under `[project.optional-dependencies]`:

| Extra    | Installs             | What it enables |
| -------- | -------------------- | --------------- |
| `images` | `pillow`             | Image handling helpers in `novakit.llm` |
| `mcp`    | (no extra runtime deps) | `novakit.mcp` client, stdio/HTTP transports, tool-name allocation |

```bash
pip install "novakit[images]"
pip install "novakit[mcp]"
pip install "novakit[images,mcp]"
```

There is no `docs` extra in the current `pyproject.toml`; documentation
tooling (MkDocs, `mkdocstrings`) is a docs-infrastructure concern handled by
the infrastructure lane, not a runtime dependency.

## For local development

```bash
git clone git@github.com:bigrivi/novakit.git
cd novakit
uv sync --extra dev
```

The `dev` extra provides `mypy`, `pytest`, `pytest-asyncio`, `pytest-cov`, and
`ruff`. The full preflight is:

```bash
uv run pytest -q --cov=novakit --cov-report=term-missing
uv run ruff check novakit tests
uv run ruff format novakit tests
uv run mypy
```

The suite enforces at least 99% line coverage for `novakit` and runs offline;
provider transports, filesystem access, and external services are replaced
with test doubles.

## Next step

Continue with the runnable [Quickstart](quickstart.md).
