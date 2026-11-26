# Changelog

All notable changes to `novakit` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Bilingual English/Chinese docs with English default and /zh/.

## [0.1.0] - 2026-11-26

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
- Offline regression suite expanded to 1024 passing tests and 3 strict xfails for
  known provider edge-case defects, with 99.43% line coverage.
- Portable compaction planning, safe splitting, tool-result snipping, summary
  generation/retry, circuit breaking, and a default `Compactor` implementation.
- Explicit `CompactionConfig`, `TokenResolver`, `ToolOutputSink`, and
  `CompactionStore` injection, plus `MemoryCompactionStore` for standalone use.
- Portable session-title prompts, message construction, cleanup, length limits,
  one-shot generation, timeout injection, and failure handling.
- Portable `novakit.tasks` runtime with executor contracts, concurrency pools,
  quotas, timeouts, cancellation, listeners, retention, and bounded output.
- Optional `novakit[mcp]` module with JSON-RPC client, stdio/HTTP transports,
  injectable client identity, and deterministic tool-name allocation.
- Portable project-instruction discovery, precedence/merge/truncation, and
  configurable untrusted-text threat screening.
- Portable in-memory agent hierarchy and task-local spawn-depth tracking.
- Portable session-fork validation/copy algorithm behind `SessionForkStore`.
- Product-neutral prompt section composition, environment, roster, and skill
  formatting helpers.
- Pure per-million-token pricing math with unknown-price semantics.
- Portable child-process execution with merged output, cancellation, and
  cross-platform process-tree termination through `novakit.tasks.process`.
- Portable skill models, `SKILL.md` scanning, path validation, and case-insensitive
  catalog through `novakit.skills`.
- Portable file checkpoint capture/restore through `novakit.checkpoints`, with an
  explicit root path or root provider and no user-directory default.
- Tagged release workflow with build/publish separation, TestPyPI-before-PyPI
  promotion, protected environments, OIDC Trusted Publishing, version checks,
  release preflight, and isolated TestPyPI install smoke.

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
