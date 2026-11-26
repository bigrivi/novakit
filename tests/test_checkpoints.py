"""Direct tests for portable checkpoints with an explicit storage root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from novakit.checkpoints import (
    ENTRIES_FILENAME,
    MAX_SNAPSHOT_BYTES,
    META_FILENAME,
    CheckpointStore,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "checkpoints")


def _begin(
    store: CheckpointStore,
    workspace: Path,
    checkpoint_id: str = "msg-1",
) -> None:
    store.begin_turn(
        session_id="s1",
        checkpoint_id=checkpoint_id,
        label="do the thing",
        workspace=str(workspace),
    )


def test_captures_pre_image_and_post_hash(
    store: CheckpointStore, workspace: Path
) -> None:
    target = workspace / "a.txt"
    target.write_text("old\n", encoding="utf-8")
    _begin(store, workspace)

    store.record_write(target, "old\n", "new\n")
    target.write_text("new\n", encoding="utf-8")

    summaries = store.list_checkpoints("s1")
    assert [item.checkpoint_id for item in summaries] == ["msg-1"]
    assert summaries[0].files == 1
    assert summaries[0].label == "do the thing"


def test_restores_first_pre_image_for_repeated_writes(
    store: CheckpointStore, workspace: Path
) -> None:
    target = workspace / "a.txt"
    target.write_text("first\n", encoding="utf-8")
    _begin(store, workspace)

    store.record_write(target, "first\n", "second\n")
    store.record_write(target, "second\n", "third\n")
    target.write_text("third\n", encoding="utf-8")

    report = store.restore_files("s1", "msg-1")

    assert report.restored == [str(target)]
    assert target.read_text(encoding="utf-8") == "first\n"


def test_restore_deletes_files_created_by_the_turn(
    store: CheckpointStore, workspace: Path
) -> None:
    target = workspace / "nested" / "new.txt"
    target.parent.mkdir()
    _begin(store, workspace)

    store.record_write(target, None, "hello\n")
    target.write_text("hello\n", encoding="utf-8")

    report = store.restore_files("s1", "msg-1")

    assert report.deleted == [str(target)]
    assert not target.exists()
    assert not target.parent.exists()


def test_binary_and_oversized_pre_images_are_reported(
    store: CheckpointStore, workspace: Path
) -> None:
    binary = workspace / "bin.dat"
    big = workspace / "big.txt"
    binary.write_bytes(b"\x00\x01")
    _begin(store, workspace)

    store.record_write(binary, "\x00binary", "x")
    store.record_write(big, "x" * (MAX_SNAPSHOT_BYTES + 1), "y")

    report = store.restore_files("s1", "msg-1")
    reasons = {item["path"]: item["reason"] for item in report.skipped}
    assert reasons[str(binary)] == "binary"
    assert reasons[str(big)] == "too-large"


def test_changed_since_requires_force(store: CheckpointStore, workspace: Path) -> None:
    target = workspace / "a.txt"
    target.write_text("old\n", encoding="utf-8")
    _begin(store, workspace)
    store.record_write(target, "old\n", "agent\n")
    target.write_text("agent\n", encoding="utf-8")
    target.write_text("human\n", encoding="utf-8")

    report = store.restore_files("s1", "msg-1")

    assert report.restored == []
    assert report.skipped == [{"path": str(target), "reason": "changed-since"}]
    assert target.read_text(encoding="utf-8") == "human\n"

    forced = store.restore_files("s1", "msg-1", force=True)
    assert forced.restored == [str(target)]
    assert target.read_text(encoding="utf-8") == "old\n"


def test_restore_refuses_paths_outside_workspace(
    store: CheckpointStore, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n", encoding="utf-8")
    directory = store.root / "s1" / "msg-1"
    directory.mkdir(parents=True)
    (directory / META_FILENAME).write_text(
        json.dumps({"checkpoint_id": "msg-1", "workspace": str(workspace)}),
        encoding="utf-8",
    )
    (directory / ENTRIES_FILENAME).write_text(
        json.dumps({"path": str(outside), "existed": False, "post_hash": "x"}) + "\n",
        encoding="utf-8",
    )

    report = store.restore_files("s1", "msg-1")

    assert report.skipped == [{"path": str(outside), "reason": "outside-workspace"}]
    assert outside.read_text(encoding="utf-8") == "keep\n"


def test_missing_checkpoint_raises(store: CheckpointStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.restore_files("s1", "missing")


def test_discard_removes_target_and_newer_checkpoints(
    store: CheckpointStore, workspace: Path
) -> None:
    target = workspace / "a.txt"
    target.write_text("a\n", encoding="utf-8")
    for checkpoint_id in ("msg-1", "msg-2", "msg-3"):
        _begin(store, workspace, checkpoint_id)
        store.record_write(target, "a\n", checkpoint_id)

    store.discard_checkpoints_from("s1", "msg-2")

    assert [item.checkpoint_id for item in store.list_checkpoints("s1")] == ["msg-1"]


def test_session_paths_returns_unique_captured_paths(
    store: CheckpointStore, workspace: Path
) -> None:
    first = workspace / "a.txt"
    second = workspace / "b.txt"
    _begin(store, workspace)
    store.record_write(first, None, "a")
    store.record_write(second, None, "b")
    store.record_write(first, "a", "aa")

    assert store.session_paths("s1") == {str(first), str(second)}


def test_explicit_root_does_not_create_user_directories(
    store: CheckpointStore,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    nova_home = tmp_path / "nova-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NOVA_HOME", str(nova_home))
    target = workspace / "a.txt"
    _begin(store, workspace)
    store.record_write(target, None, "new")

    assert not home.exists()
    assert not nova_home.exists()


def test_callable_root_and_missing_turn_identifiers(tmp_path: Path) -> None:
    root = tmp_path / "first"
    store = CheckpointStore(lambda: root)

    assert store.begin_turn("", "msg", "label", None) is None
    assert store.begin_turn("s1", "", "label", None) is None
    store.record_write(tmp_path / "ignored.txt", None, "new")
    assert store.list_checkpoints("missing") == []

    root = tmp_path / "second"
    assert store.root == root


def test_malformed_metadata_is_ignored(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    directory = store.root / "s1" / "msg-1"
    directory.mkdir(parents=True)
    (directory / META_FILENAME).write_text("not json", encoding="utf-8")

    assert store.list_checkpoints("s1") == []


def test_restore_handles_already_missing_created_file(
    store: CheckpointStore, workspace: Path
) -> None:
    missing = workspace / "missing.txt"
    directory = store.root / "s1" / "msg-1"
    directory.mkdir(parents=True)
    (directory / META_FILENAME).write_text(
        json.dumps({"checkpoint_id": "msg-1", "workspace": str(workspace)}),
        encoding="utf-8",
    )
    (directory / ENTRIES_FILENAME).write_text(
        "\n".join(
            [
                json.dumps({"path": str(missing), "existed": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = store.restore_files("s1", "msg-1")

    assert report.skipped == []
    assert report.deleted == [str(missing)]


def test_malformed_and_partial_entries_are_ignored(
    store: CheckpointStore, workspace: Path
) -> None:
    target = workspace / "a.txt"
    directory = store.root / "s1" / "msg-1"
    directory.mkdir(parents=True)
    (directory / META_FILENAME).write_text(
        json.dumps({"checkpoint_id": "msg-1", "workspace": str(workspace)}),
        encoding="utf-8",
    )
    (directory / ENTRIES_FILENAME).write_text(
        "\n".join(
            [
                "not json",
                "[]",
                json.dumps({"path": ""}),
                json.dumps({"path": str(target), "post_hash": "update-only"}),
                json.dumps({"path": str(target), "existed": True, "content": "old"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert store.session_paths("s1") == {str(target)}
    assert store.list_checkpoints("s1")[0].files == 1


def test_discard_unknown_and_remove_missing_are_noops(store: CheckpointStore) -> None:
    store.discard_checkpoints_from("s1", "missing")
    store.remove_session("s1")
