"""Portable file checkpoints with an explicitly supplied storage root.

The framework never chooses a home directory. Hosts provide either a fixed root
path or a callable that resolves one, then decide where checkpoint data lives.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MAX_SNAPSHOT_BYTES = 256 * 1024
MAX_CHECKPOINTS_PER_SESSION = 50
META_FILENAME = "meta.json"
ENTRIES_FILENAME = "entries.jsonl"

RootProvider = Callable[[], Path]


@dataclass
class CheckpointSummary:
    """One rewind point as exposed to a host client."""

    session_id: str
    checkpoint_id: str
    label: str
    created_at: int
    files: int
    skipped: int


@dataclass
class RestoreReport:
    """Files restored, deleted, or skipped during a rewind."""

    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


@dataclass
class TurnRecorder:
    """Pre-images recorded for one turn on a specific checkpoint store."""

    store: CheckpointStore
    session_id: str
    checkpoint_id: str
    label: str
    workspace: str
    recorded: set[str] = field(default_factory=set)

    @property
    def directory(self) -> Path:
        """Directory containing this turn's checkpoint files."""
        return self.store.root / self.session_id / self.checkpoint_id

    def already_recorded(self, path: Path) -> bool:
        """Return whether this turn already captured a pre-image for *path*."""
        return str(path) in self.recorded


class CheckpointStore:
    """Capture, list, restore, and prune file checkpoints under an explicit root."""

    def __init__(self, root: Path | RootProvider) -> None:
        """Create a store without performing filesystem I/O.

        Args:
            root: Fixed checkpoint directory or a callable resolving it. The
                callable form lets a host change its configured home at runtime.
        """
        if callable(root):
            self._root_provider = root
        else:
            fixed_root = root
            self._root_provider = lambda: fixed_root
        self._recorder: ContextVar[TurnRecorder | None] = ContextVar(
            f"novakit-checkpoint-{id(self)}", default=None
        )

    @property
    def root(self) -> Path:
        """Resolve the host-supplied checkpoint root."""
        return Path(self._root_provider())

    def begin_turn(
        self,
        session_id: str,
        checkpoint_id: str,
        label: str,
        workspace: str | None,
    ) -> TurnRecorder | None:
        """Start capturing writes for one turn.

        Args:
            session_id: Owning session identifier.
            checkpoint_id: User message identifier used as the rewind point.
            label: Human-readable turn label.
            workspace: Workspace boundary used during restore.

        Returns:
            The active recorder, or None when session/checkpoint identifiers are
            missing.
        """
        if not session_id or not checkpoint_id:
            return None
        recorder = TurnRecorder(
            store=self,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            label=label.strip()[:200],
            workspace=workspace or "",
        )
        self._recorder.set(recorder)
        return recorder

    def current_recorder(self) -> TurnRecorder | None:
        """Return the recorder for the current task, if any."""
        return self._recorder.get()

    def record_write(self, path: Path, old_text: str | None, new_text: str) -> None:
        """Capture a file pre-image before the host applies *new_text*.

        Args:
            path: File the host is about to write.
            old_text: Previous text, or None when the file did not exist.
            new_text: New text used to compute the post-image hash.
        """
        recorder = self.current_recorder()
        if recorder is None:
            return
        if recorder.already_recorded(path):
            self._append_entry(
                recorder,
                {
                    "path": str(path),
                    "post_hash": hashlib.sha256(new_text.encode("utf-8")).hexdigest(),
                },
            )
            return
        recorder.recorded.add(str(path))

        entry: dict[str, object] = {
            "path": str(path),
            "existed": old_text is not None,
            "post_hash": hashlib.sha256(new_text.encode("utf-8")).hexdigest(),
        }
        if old_text is None:
            entry["content"] = None
        elif "\x00" in old_text:
            entry["skipped"] = "binary"
        elif len(old_text.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            entry["skipped"] = "too-large"
        else:
            entry["content"] = old_text

        try:
            self._append_entry(recorder, entry)
        except OSError:
            log.exception("Checkpoint write failed for %s", path)

    def list_checkpoints(self, session_id: str) -> list[CheckpointSummary]:
        """List a session's checkpoints, newest first.

        Args:
            session_id: Session whose checkpoints should be listed.

        Returns:
            Summary records ordered by creation time.
        """
        directory = self.root / session_id
        try:
            children = [item for item in directory.iterdir() if item.is_dir()]
        except OSError:
            return []

        summaries: list[tuple[int, CheckpointSummary]] = []
        for child in children:
            meta = _read_meta(child)
            if meta is None:
                continue
            entries = _read_entries(child)
            try:
                tie_break = child.stat().st_mtime_ns
            except OSError:
                tie_break = 0
            summaries.append(
                (
                    tie_break,
                    CheckpointSummary(
                        session_id=session_id,
                        checkpoint_id=str(meta.get("checkpoint_id") or child.name),
                        label=str(meta.get("label") or ""),
                        created_at=int(str(meta.get("created_at") or 0)),
                        files=len(entries),
                        skipped=sum(1 for entry in entries if entry.get("skipped")),
                    ),
                )
            )
        summaries.sort(key=lambda pair: (pair[1].created_at, pair[0]), reverse=True)
        return [summary for _, summary in summaries]

    def restore_files(
        self,
        session_id: str,
        checkpoint_id: str,
        *,
        force: bool = False,
    ) -> RestoreReport:
        """Restore captured files while respecting workspace and conflict checks.

        Args:
            session_id: Owning session identifier.
            checkpoint_id: Rewind point to restore.
            force: Restore even when a file changed after the agent wrote it.

        Returns:
            A report of restored, deleted, and skipped paths.

        Raises:
            FileNotFoundError: When the checkpoint does not exist.
        """
        directory = self.root / session_id / checkpoint_id
        meta = _read_meta(directory)
        if meta is None:
            raise FileNotFoundError(f"No checkpoint {checkpoint_id} for {session_id}")

        workspace = Path(str(meta.get("workspace") or ""))
        report = RestoreReport()
        for entry in _read_entries(directory):
            path_text = str(entry.get("path") or "")
            path = Path(path_text)
            if not path_text:
                report.skipped.append({"path": path_text, "reason": "no-path"})
                continue
            if not workspace.is_dir() or not _is_inside(path, workspace):
                report.skipped.append(
                    {"path": path_text, "reason": "outside-workspace"}
                )
                continue

            reason = str(entry.get("skipped") or "")
            if reason:
                report.skipped.append({"path": path_text, "reason": reason})
                continue
            if not force and not _still_matches(path, entry):
                report.skipped.append({"path": path_text, "reason": "changed-since"})
                continue

            try:
                if entry.get("existed"):
                    _write_atomic(path, str(entry.get("content") or ""))
                    report.restored.append(path_text)
                elif path.exists():
                    path.unlink()
                    _prune_empty_parents(path.parent, workspace)
                    report.deleted.append(path_text)
                else:
                    report.deleted.append(path_text)
            except OSError as error:
                log.exception("Restore failed for %s", path)
                report.skipped.append({"path": path_text, "reason": f"error: {error}"})
        return report

    def discard_checkpoints_from(self, session_id: str, checkpoint_id: str) -> None:
        """Delete a checkpoint and every newer checkpoint for the session.

        Args:
            session_id: Owning session identifier.
            checkpoint_id: Oldest rewind point to discard.
        """
        summaries = self.list_checkpoints(session_id)
        index = next(
            (
                position
                for position, item in enumerate(summaries)
                if item.checkpoint_id == checkpoint_id
            ),
            None,
        )
        if index is None:
            return
        for stale in summaries[: index + 1]:
            _remove(self.root / session_id / stale.checkpoint_id)

    def prune_session(self, session_id: str) -> None:
        """Keep only the newest configured number of checkpoints for a session.

        Args:
            session_id: Session whose old checkpoints should be pruned.
        """
        directory = self.root / session_id
        try:
            children = sorted(
                (item for item in directory.iterdir() if item.is_dir()),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        for stale in children[MAX_CHECKPOINTS_PER_SESSION:]:
            _remove(stale)

    def remove_session(self, session_id: str) -> None:
        """Delete all checkpoints owned by a session.

        Args:
            session_id: Session whose checkpoint directory should be removed.
        """
        _remove(self.root / session_id)

    def session_paths(self, session_id: str) -> set[str]:
        """Return every file path recorded by any checkpoint for a session.

        Args:
            session_id: Session whose touched paths should be returned.

        Returns:
            Absolute path strings captured by the session.
        """
        directory = self.root / session_id
        try:
            children = [item for item in directory.iterdir() if item.is_dir()]
        except OSError:
            return set()
        paths: set[str] = set()
        for child in children:
            for entry in _read_entries(child):
                path = str(entry.get("path") or "")
                if path:
                    paths.add(path)
        return paths

    def _append_entry(self, recorder: TurnRecorder, entry: dict[str, object]) -> None:
        directory = recorder.directory
        directory.mkdir(parents=True, exist_ok=True)
        meta = directory / META_FILENAME
        if not meta.exists():
            meta.write_text(
                json.dumps(
                    {
                        "session_id": recorder.session_id,
                        "checkpoint_id": recorder.checkpoint_id,
                        "label": recorder.label,
                        "workspace": recorder.workspace,
                        "created_at": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.prune_session(recorder.session_id)
        with (directory / ENTRIES_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_meta(directory: Path) -> dict[str, object] | None:
    try:
        raw = (directory / META_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Ignoring malformed checkpoint meta at %s", directory)
        return None
    return payload if isinstance(payload, dict) else None


def _read_entries(directory: Path) -> list[dict[str, object]]:
    """Merge an append-only log into one entry per path, first-seen order."""
    try:
        lines = (directory / ENTRIES_FILENAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    merged: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Ignoring malformed checkpoint entry at %s", directory)
            continue
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "")
        if not path:
            continue
        if "existed" not in entry:
            if path in merged:
                merged[path]["post_hash"] = entry.get("post_hash")
            continue
        if path not in merged:
            order.append(path)
        merged[path] = entry
    return [merged[path] for path in order]


def _is_inside(path: Path, workspace: Path) -> bool:
    try:
        path.resolve().relative_to(workspace.resolve())
    except (OSError, ValueError):
        return False
    return True


def _still_matches(path: Path, entry: dict[str, object]) -> bool:
    """Return whether the file still holds the agent's final write."""
    expected = str(entry.get("post_hash") or "")
    if not expected:
        return not entry.get("existed")
    try:
        current = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        current = ""
    return current == expected


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.novakit-restore")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _prune_empty_parents(directory: Path, workspace: Path) -> None:
    current = directory
    while current != workspace and _is_inside(current, workspace):
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remove(directory: Path) -> None:
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        pass
    except OSError:
        log.exception("Could not remove checkpoint directory %s", directory)


__all__ = [
    "ENTRIES_FILENAME",
    "MAX_CHECKPOINTS_PER_SESSION",
    "MAX_SNAPSHOT_BYTES",
    "META_FILENAME",
    "CheckpointStore",
    "CheckpointSummary",
    "RestoreReport",
    "RootProvider",
    "TurnRecorder",
]
