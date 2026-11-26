"""Shared local teaching servers for examples 12-15.

These helpers run tiny ``aiohttp`` mock servers (or a stdio script) on
127.0.0.1 with ephemeral ports. No external network, no API keys, fully
deterministic. Numbered examples only configure transports, providers,
and clients, then print results; all wire-protocol stubbing lives here.
"""

from __future__ import annotations
