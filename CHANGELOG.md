# Changelog

All notable changes to `novakit` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Standalone repository scaffolding, CI, wheel smoke test, and handoff documentation.
- `CoreConfig` protocol for the narrow runtime configuration surface.
- `InMemoryApprovalManager` with heartbeat, abort, and session-scoped grants.
- `InMemorySession` for default per-agent memory persistence.
- `Agent`, `ToolInvoker`, streaming events, tool behaviors, and LLM providers.
- Host ports for session persistence, compaction, approvals, and checkpoints.
- Expanded public API docstrings for events, sessions, tools, providers, and
  runtime protocols.
- `pytest-cov` development dependency and a 99% line-coverage CI gate.
- Offline regression suite expanded to 826 passing tests and 3 strict xfails for
  known provider edge-case defects, with 99.98% line coverage.

### Changed

- The core runtime depends only on `CoreConfig`; `AgentConfig` remains a
  backward-compatible superset for Nova consumers.
- Configuration ownership decided (option A): `AgentConfig` keeps its 11-field
  order frozen in `0.x` for host compatibility; the standalone core reads only
  the four `CoreConfig` fields and ignores the rest. New host fields must not
  be added to `AgentConfig`; removing or reordering fields is a major change.
- `ToolGuardrails` default `read_tools` keeps the historical host tool names
  for compatibility (frozen in `0.x`); hosts should pass `read_tools`
  explicitly, e.g. derived from `Tool.read_only`.
- `ToolRegistry.from_mcp` comments reworded to neutral host phrasing; no
  behavior change.
- Docs: README gains a core/host field table, a `CoreConfig`-shaped host
  config example, and a `ToolGuardrails(read_tools=...)` example.
- Tests: positional `AgentConfig` order lock plus a core-ignores-host-fields
  runtime test.
