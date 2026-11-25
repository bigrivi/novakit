"""Offline tests for novakit.llm.images.

Data-URL parsing, size estimation, downscaling (via an in-memory PIL stand-in
so no image library or fixture file is needed), and the request image budget.
"""

from __future__ import annotations

import base64
import sys
import types

import pytest

from novakit.llm import images
from novakit.llm.images import (
    BOUND_TARGET_BYTES,
    BOUND_TRIGGER_BYTES,
    DEFAULT_IMAGE_MIME,
    DOWNSCALE_BYTES,
    DOWNSCALE_MAX_EDGE,
    IMAGE_REMOVED,
    bound_images,
    downscale_image,
    image_data_url,
    image_size_bytes,
    parse_image_ref,
)
from novakit.llm.provider import Message


def _ref(payload: bytes, mime: str = "image/png") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _large_ref() -> str:
    return _ref(b"x" * (DOWNSCALE_BYTES + 16))


def _ten_byte_image() -> str:
    return _ref(b"0123456789")


class TestParseImageRef:
    def test_splits_a_data_url(self) -> None:
        assert parse_image_ref("data:image/jpeg;base64,QUJD") == ("image/jpeg", "QUJD")

    def test_tolerates_parameters_after_the_mime(self) -> None:
        mime, payload = parse_image_ref("data:image/webp;charset=utf-8;base64,QUJD")

        assert (mime, payload) == ("image/webp", "QUJD")

    def test_reads_bare_base64_as_png(self) -> None:
        assert parse_image_ref("QUJD") == (DEFAULT_IMAGE_MIME, "QUJD")

    def test_missing_mime_falls_back(self) -> None:
        mime, payload = parse_image_ref("data:;base64,QUJD")

        assert (mime, payload) == (DEFAULT_IMAGE_MIME, "QUJD")


class TestImageDataUrl:
    def test_round_trips(self) -> None:
        assert (
            image_data_url("data:image/gif;base64,R0lG") == "data:image/gif;base64,R0lG"
        )

    def test_wraps_bare_base64(self) -> None:
        assert image_data_url("QUJD") == "data:image/png;base64,QUJD"

    def test_rewrites_legacy_mime_params(self) -> None:
        ref = image_data_url("data:image/webp;charset=utf-8;base64,QUJD")

        assert ref == "data:image/webp;base64,QUJD"


class TestImageSizeBytes:
    def test_decodes_from_the_base64_length(self) -> None:
        assert image_size_bytes("data:image/png;base64,QUJD") == 3

    def test_ignores_padding(self) -> None:
        assert image_size_bytes("QUI=") == 2

    def test_empty_payload(self) -> None:
        assert image_size_bytes("data:image/png;base64,") == 0

    def test_bare_base64(self) -> None:
        assert image_size_bytes(base64.b64encode(b"0123456789").decode()) == 10


class _FakeResampling:
    LANCZOS = 1


def _install_fake_pil(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "RGB",
    info: dict[str, object] | None = None,
    fail_open: bool = False,
) -> dict[str, object]:
    """Provide a minimal PIL stand-in recording how it was used."""
    calls: dict[str, object] = {}

    class _Image:
        Resampling = _FakeResampling

        def __init__(self) -> None:
            self.mode = mode
            self.info = dict(info or {})
            self.size = (3000, 2000)

        def __enter__(self) -> _Image:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def thumbnail(self, size: tuple[int, int], resample: object) -> None:
            calls["thumbnail"] = (size, resample)
            self.size = (1000, 667)

        def convert(self, colour: str) -> _Image:
            calls["convert"] = colour
            return self

        def save(
            self,
            buffer: object,
            *,
            format: str,
            quality: int | None = None,
            optimize: bool | None = None,
        ) -> None:
            calls["save"] = {
                "format": format,
                "quality": quality,
                "optimize": optimize,
            }
            assert hasattr(buffer, "write")
            buffer.write(b"FAKE-IMAGE-BYTES")  # type: ignore[union-attr]

        @staticmethod
        def open(buffer: object) -> _Image:
            assert hasattr(buffer, "read")
            calls["raw"] = buffer.read()  # type: ignore[union-attr]
            if fail_open:
                raise OSError("cannot identify image file")
            return _Image()

    pil = types.ModuleType("PIL")
    pil.Image = _Image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil)
    return calls


class TestDownscaleImage:
    def test_a_small_image_is_left_alone(self) -> None:
        small = _ten_byte_image()
        assert image_size_bytes(small) < DOWNSCALE_BYTES

        assert downscale_image(small) == small

    def test_photo_becomes_jpeg_at_provider_edge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_fake_pil(monkeypatch, mode="RGB")

        smaller = downscale_image(_large_ref())

        assert smaller.startswith("data:image/jpeg;base64,")
        assert calls["thumbnail"] == ((DOWNSCALE_MAX_EDGE, DOWNSCALE_MAX_EDGE), 1)
        assert calls["convert"] == "RGB"
        assert calls["save"] == {
            "format": "JPEG",
            "quality": images.DOWNSCALE_QUALITY,
            "optimize": True,
        }

    def test_transparency_survives_as_png(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_fake_pil(monkeypatch, mode="RGBA")

        smaller = downscale_image(_large_ref())

        assert smaller.startswith("data:image/png;base64,")
        assert calls["save"]["format"] == "PNG"  # type: ignore[index]

    def test_la_mode_keeps_alpha(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_pil(monkeypatch, mode="LA")

        assert downscale_image(_large_ref()).startswith("data:image/png;base64,")

    def test_transparency_info_keeps_alpha(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_pil(monkeypatch, mode="P", info={"transparency": 1})

        assert downscale_image(_large_ref()).startswith("data:image/png;base64,")

    def test_an_unreadable_payload_is_returned_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_pil(monkeypatch, fail_open=True)
        ref = _large_ref()

        assert downscale_image(ref) == ref

    def test_missing_pil_returns_ref_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "PIL", None)
        ref = _large_ref()

        assert downscale_image(ref) == ref


class TestBoundImages:
    def test_a_request_under_the_trigger_is_untouched(self) -> None:
        messages = [
            Message(role="user", content="a", images=[_ten_byte_image()]),
            Message(role="user", content="b", images=[_ten_byte_image()]),
        ]

        assert bound_images(messages) is messages

    def test_trigger_boundary_is_untouched(self) -> None:
        messages = [Message(role="user", content="a", images=[_ten_byte_image()])]

        assert bound_images(messages, trigger=10, target=5) is messages

    def test_the_oldest_images_are_dropped_first(self) -> None:
        messages = [
            Message(role="user", content=f"turn {index}", images=[_ten_byte_image()])
            for index in range(4)
        ]

        bounded = bound_images(messages, trigger=25, target=10)

        assert [m.images is None for m in bounded] == [True, True, False, False]
        assert bounded[0].content == f"turn 0\n\n{IMAGE_REMOVED}"
        assert bounded[1].content == f"turn 1\n\n{IMAGE_REMOVED}"
        assert bounded[2].content == "turn 2"
        assert bounded[3].content == "turn 3"

    def test_a_partly_cut_message_keeps_its_newest_image(self) -> None:
        original = [_ten_byte_image() for _ in range(3)]
        messages = [Message(role="user", content="three pictures", images=original)]

        bounded = bound_images(messages, trigger=15, target=5)

        assert bounded[0].images == [original[2]]
        assert bounded[0].content.count(IMAGE_REMOVED) == 1
        assert messages[0].images == original
        assert messages[0].content == "three pictures"

    def test_messages_without_images_survive_a_cut(self) -> None:
        original = [_ten_byte_image()] * 4
        messages = [
            Message(role="user", content="plain", images=None),
            Message(role="user", content="pics", images=list(original)),
        ]

        bounded = bound_images(messages, trigger=25, target=10)

        assert bounded[0] is messages[0]
        assert bounded[1].images == original[2:]
        assert IMAGE_REMOVED in bounded[1].content

    def test_cut_can_consume_every_image(self) -> None:
        messages = [
            Message(role="user", content=f"turn {i}", images=[_ten_byte_image()])
            for i in range(3)
        ]

        bounded = bound_images(messages, trigger=25, target=5)

        assert all(m.images is None for m in bounded)
        assert all(IMAGE_REMOVED in m.content for m in bounded)

    def test_custom_trigger_and_target_constants(self) -> None:
        assert (BOUND_TRIGGER_BYTES, BOUND_TARGET_BYTES) == (
            25 * 1024 * 1024,
            15 * 1024 * 1024,
        )

    def test_image_removed_placeholder_names_the_fix(self) -> None:
        assert "tool" in IMAGE_REMOVED
