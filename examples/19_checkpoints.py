"""Checkpoints: pre-image capture, restore, and discard semantics.

What you'll learn:
  - How begin_turn + record_write capture pre-images for restore.
  - How restore handles edited files, created files, and human edits.
  - How discard_checkpoints_from drops a checkpoint and newer ones.

Run:
    python examples/19_checkpoints.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from novakit.agent import AgentPorts
from novakit.checkpoints import CheckpointStore


def _begin(
    store: CheckpointStore, workspace: Path, checkpoint_id: str = "msg-1"
) -> None:
    store.begin_turn(
        session_id="s1",
        checkpoint_id=checkpoint_id,
        label=f"turn {checkpoint_id}",
        workspace=str(workspace),
    )


def main() -> None:
    # Prepare data: workspace + checkpoint store both live in tempdir, so
    # every write and restore stays local and disposable.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        workspace.mkdir()
        store = CheckpointStore(root / "checkpoints")

        # Call SDK: begin_turn is the same hook an Agent calls via AgentPorts
        # at the start of each turn.
        ports = AgentPorts(begin_turn=store.begin_turn)
        print(f"begin_turn hook wired: {ports.begin_turn is not None}")
        print(
            "begin_turn usage: ports.begin_turn(session, checkpoint, label, workspace)"
        )
        ports.begin_turn("s1", "msg-1", "edit a.txt", str(workspace))

        # Prepare data: capture a pre-image, then simulate the agent edit.
        target = workspace / "a.txt"
        target.write_text("original\n", encoding="utf-8")
        store.record_write(target, "original\n", "edited\n")
        target.write_text("edited\n", encoding="utf-8")
        print(f"before restore: {target.read_text(encoding='utf-8')!r}")

        # Call SDK: restore brings back the original text.
        report = store.restore_files("s1", "msg-1")
        print(f"after restore: {target.read_text(encoding='utf-8')!r}")
        print(f"restore report: restored={report.restored}")
        summaries = store.list_checkpoints("s1")
        print("checkpoint summaries:")
        for summary in summaries:
            print(f"  {summary.checkpoint_id}: files={summary.files}")

        # Call SDK: repeated writes keep the first pre-image, so restoring
        # msg-2 still returns to "original", not an intermediate edit.
        store.begin_turn("s1", "msg-2", "two writes", str(workspace))
        store.record_write(target, "original\n", "second\n")
        store.record_write(target, "second\n", "third\n")
        target.write_text("third\n", encoding="utf-8")
        print(f"before second restore: {target.read_text(encoding='utf-8')!r}")
        second = store.restore_files("s1", "msg-2")
        print(f"after second restore: {target.read_text(encoding='utf-8')!r}")
        print(f"second restore report: restored={second.restored}")

        # Call SDK: files created during the turn are deleted on restore,
        # which is how the workspace returns to its pre-turn shape.
        _begin(store, workspace, "msg-3")
        created = workspace / "nested" / "new.txt"
        created.parent.mkdir(parents=True)
        store.record_write(created, None, "hello\n")
        created.write_text("hello\n", encoding="utf-8")
        created_report = store.restore_files("s1", "msg-3")
        print(f"created-file report: deleted={created_report.deleted}")
        print(f"created file still exists: {created.exists()}")

        # Call SDK: a human edit after the agent blocks restore (reported as
        # skipped), unless force=True explicitly overrides the guard.
        _begin(store, workspace, "msg-4")
        store.record_write(target, "original\n", "agent\n")
        target.write_text("agent\n", encoding="utf-8")
        target.write_text("human\n", encoding="utf-8")
        guarded = store.restore_files("s1", "msg-4")
        print(f"guarded restore: restored={guarded.restored} skipped={guarded.skipped}")
        print(f"content after guarded restore: {target.read_text(encoding='utf-8')!r}")
        forced = store.restore_files("s1", "msg-4", force=True)
        print(f"forced restore: restored={forced.restored}")
        print(f"content after forced restore: {target.read_text(encoding='utf-8')!r}")

        # Call SDK: discard removes the target checkpoint and every newer
        # one; newest-first order is msg-4..msg-1, so dropping from msg-3
        # keeps only msg-2 and msg-1.
        print(
            f"checkpoints before discard: {[s.checkpoint_id for s in store.list_checkpoints('s1')]}"
        )
        store.discard_checkpoints_from("s1", "msg-3")
        remaining = [s.checkpoint_id for s in store.list_checkpoints("s1")]
        print(f"checkpoints after discarding from msg-3: {remaining}")
        print(f"remaining session paths: {store.session_paths('s1')}")


if __name__ == "__main__":
    main()
