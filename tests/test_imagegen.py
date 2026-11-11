"""Tests for `manifoldx.imagegen`: the two services, the budget log and the prompts.

Every server here is a fake on 127.0.0.1; nothing reaches the network."""

from __future__ import annotations

import base64
import csv
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from manifoldx.devchannel import png
from manifoldx.imagegen import (BudgetExceeded, HttpImage, ImageModel, OpenRouterImage,
                                _decode_png, albedo, blueprint, edit, emission_map,
                                normal_map, roughness_map)


def _image() -> np.ndarray:
    img = np.zeros((4, 5, 3), np.uint8)
    img[0, 0] = (1, 2, 3)
    img[1, 2] = (200, 100, 50)
    img[3, 4] = (255, 255, 255)
    return img


def _data_url(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png(img)).decode("ascii")


def _answer(url: str, cost: float = 0.039) -> dict:
    return {"choices": [{"message": {"content": "", "images": [{"image_url": {"url": url}}]}}],
            "usage": {"cost": cost}}


class _Fake:
    """A small HTTP server whose answer each test chooses."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reply = lambda path, body, n: (200, {})
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({
                    "path": self.path,
                    "body": json.loads(raw),
                    "authorization": self.headers.get("Authorization"),
                    "content_type": self.headers.get("Content-Type"),
                })
                status, payload = outer.reply(self.path, outer.requests[-1]["body"],
                                              len(outer.requests))
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    fake = _Fake()
    try:
        yield fake
    finally:
        fake.close()


# ----------------------------------------------------------------------------- HTTP and PNG

def _filtered_png(img: np.ndarray, filters: list[int]) -> bytes:
    """A non-interlaced PNG of `img` with one filter type per row, encoded by hand."""
    h, w, channels = img.shape
    bpp = channels
    previous = np.zeros(w * channels, np.uint8)
    rows = bytearray()
    for y in range(h):
        current = np.ascontiguousarray(img[y]).reshape(-1)
        kind = filters[y % len(filters)]
        out = bytearray()
        for x in range(len(current)):
            a = int(current[x - bpp]) if x >= bpp else 0
            b = int(previous[x])
            c = int(previous[x - bpp]) if x >= bpp else 0
            if kind == 0:
                value = int(current[x])
            elif kind == 1:
                value = (int(current[x]) - a) & 0xFF
            elif kind == 2:
                value = (int(current[x]) - b) & 0xFF
            elif kind == 3:
                value = (int(current[x]) - (a + b) // 2) & 0xFF
            else:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                value = (int(current[x]) - pred) & 0xFF
            out.append(value)
        rows.append(kind)
        rows.extend(out)
        previous = current

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body)))

    colour = 2 if channels == 3 else 6
    header = struct.pack(">IIBBBBB", w, h, 8, colour, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b""))


def test_decodes_rgb_and_rgba_with_every_filter():
    rgb = _image()
    rgba = np.concatenate([rgb, np.full((4, 5, 1), 128, np.uint8)], axis=2)
    for img in (rgb, rgba):
        for filters in ([0], [1], [2], [3], [4], [0, 1, 2, 3, 4]):
            got = _decode_png(_filtered_png(img, filters))
            assert got.shape == img.shape
            np.testing.assert_array_equal(got, img)


def test_decoder_refuses_what_it_cannot_read():
    with pytest.raises(ValueError):
        _decode_png(b"not a png")
    # an 8-bit grey PNG (colour type 0)
    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body)))
    grey = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x80")) + chunk(b"IEND", b""))
    with pytest.raises(ValueError):
        _decode_png(grey)


def test_request_shape_answer_and_cost_log(server, tmp_path):
    image = _image()
    server.reply = lambda path, body, n: (200, _answer(_data_url(image)))
    log = tmp_path / "cost.csv"
    model = OpenRouterImage("google/gemini-3-pro-image-preview", key="sk-test",
                            log=log, url=server.url + "/api/v1/chat/completions")
    reference = np.full((2, 2, 3), 255, np.uint8)

    got = model.generate("a wheel seen from the front", images=(reference,), aspect="16:9",
                         size="2K", out_name="wheel_front.png")

    assert server.requests and server.requests[0]["path"] == "/api/v1/chat/completions"
    sent = server.requests[0]
    assert sent["authorization"] == "Bearer sk-test"
    assert sent["content_type"] == "application/json"
    body = sent["body"]
    assert body["model"] == "google/gemini-3-pro-image-preview"
    assert body["modalities"] == ["image", "text"]
    assert body["image_config"] == {"aspect_ratio": "16:9", "image_size": "2K"}
    assert body["usage"] == {"include": True}
    content = body["messages"][0]["content"]
    assert content[-1] == {"type": "text", "text": "a wheel seen from the front"}
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    sent_reference = _decode_png(base64.b64decode(
        content[0]["image_url"]["url"].split(",", 1)[1]))
    np.testing.assert_array_equal(sent_reference, reference)
    np.testing.assert_array_equal(got, image)
    assert model.cost == pytest.approx(0.039)

    rows = list(csv.reader(log.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    assert rows[0][0].endswith("Z") and rows[0][0][:4].isdigit()
    assert rows[0][1] == "google/gemini-3-pro-image-preview"
    assert rows[0][2] == "wheel_front.png"
    assert float(rows[0][3]) == pytest.approx(0.039)


def test_a_text_only_prompt_sends_a_string(server):
    server.reply = lambda path, body, n: (200, _answer(_data_url(_image())))
    model = OpenRouterImage("m", key="k", url=server.url)
    model.generate("plain prompt")
    assert server.requests[0]["body"]["messages"][0]["content"] == "plain prompt"


def test_retries_without_image_config_when_the_model_refuses_it(server):
    image = _image()

    def reply(path, body, n):
        if n == 1:
            return 400, {"error": {"message": "image_config is not supported"}}
        return 200, _answer(_data_url(image))

    server.reply = reply
    model = OpenRouterImage("m", key="k", url=server.url)
    got = model.generate("a wheel", aspect="9:16", size="1K")
    assert len(server.requests) == 2
    assert "image_config" in server.requests[0]["body"]
    assert "image_config" not in server.requests[1]["body"]
    np.testing.assert_array_equal(got, image)


def test_budget_refuses_before_the_call(server, tmp_path):
    server.reply = lambda path, body, n: (200, _answer(_data_url(_image()), cost=0.25))
    log = tmp_path / "cost.csv"
    log.write_text("2026-01-01T00:00:00Z,m,old.png,0.120000\n", encoding="utf-8")
    model = OpenRouterImage("m", key="k", log=log, budget=0.10, url=server.url)

    with pytest.raises(BudgetExceeded):
        model.generate("a wheel")

    assert server.requests == []
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


def test_budget_allows_a_call_under_it_and_counts_the_new_cost(server, tmp_path):
    server.reply = lambda path, body, n: (200, _answer(_data_url(_image()), cost=0.03))
    log = tmp_path / "cost.csv"
    log.write_text("2026-01-01T00:00:00Z,m,old.png,0.040000\n", encoding="utf-8")
    model = OpenRouterImage("m", key="k", log=log, budget=0.10, url=server.url)
    model.generate("a wheel", out_name="new.png")
    assert len(server.requests) == 1
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2


def test_a_budget_needs_a_log():
    with pytest.raises(ValueError):
        OpenRouterImage("m", key="k", budget=1.0)


def test_the_key_never_appears_in_an_error(server):
    server.reply = lambda path, body, n: (401, {"error": "the key sk-secret is wrong"})
    model = OpenRouterImage("m", key="sk-secret", url=server.url)
    with pytest.raises(RuntimeError) as info:
        model.generate("a wheel")
    assert "sk-secret" not in str(info.value)
    assert "***" in str(info.value)


def test_a_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterImage("m")


def test_the_key_file_is_read(server, tmp_path):
    server.reply = lambda path, body, n: (200, _answer(_data_url(_image())))
    key_file = tmp_path / "key"
    key_file.write_text("sk-from-file\n", encoding="utf-8")
    model = OpenRouterImage("m", key_file=key_file, url=server.url)
    model.generate("a wheel")
    assert server.requests[0]["authorization"] == "Bearer sk-from-file"


# ----------------------------------------------------------------------------- local service

def test_http_image_posts_the_klein_shape(server, tmp_path):
    image = _image()
    server.reply = lambda path, body, n: (200, {"image": _data_url(image)})
    token = tmp_path / "token"
    token.write_text("tok-123\n", encoding="utf-8")
    model = HttpImage(server.url, token_file=token)

    got = model.generate("a wheel", images=(image,), aspect="3:4", size="1K", seed=7)

    assert server.requests[0]["path"] == "/generate"
    assert server.requests[0]["authorization"] == "Bearer tok-123"
    body = server.requests[0]["body"]
    assert body["prompt"] == "a wheel"
    assert body["refs"] and body["refs"][0].startswith("data:image/png;base64,")
    assert (body["width"], body["height"]) == (768, 1024)
    assert body["seed"] == 7
    np.testing.assert_array_equal(got, image)
    assert model.cost == 0.0
    assert model.name == "klein"


def test_http_image_never_puts_its_token_in_an_error(server, tmp_path):
    server.reply = lambda path, body, n: (500, {"error": "token tok-123 rejected"})
    token = tmp_path / "token"
    token.write_text("tok-123", encoding="utf-8")
    model = HttpImage(server.url, token_file=token)
    with pytest.raises(RuntimeError) as info:
        model.generate("a wheel")
    assert "tok-123" not in str(info.value)


# ----------------------------------------------------------------------------- prompts

def test_blueprint_names_every_palette_colour():
    text = blueprint("wheel", "front",
                             {"rim": (200, 0, 0), "tyre": (20, 20, 20), "hub": "white"})
    assert "#c80000" in text
    assert "#141414" in text
    assert "hub white" in text
    assert "Orthographic front view of the wheel" in text
    assert "no text" in text


def test_map_prompts_state_their_convention():
    assert "128, 128, 255" in normal_map()
    assert "white is rough" in roughness_map({"paint": 0.4, "glass": 0.05})
    assert "paint 0.40" in roughness_map({"paint": 0.4})
    assert "headlamp lit" in emission_map({"headlamp": True, "body": False})
    assert "same canvas" in edit("paint it red", "the wheel").lower()
    for text in (normal_map(), roughness_map(["a"]),
                 emission_map(["a"]), edit("x", "y"),
                 albedo("front", {"#c80000": "paint"})):
        assert "no text" in text.lower()


def test_albedo_says_what_each_colour_means():
    text = albedo("left", {"#c80000": "body paint", "chrome": (200, 200, 200)})
    assert "#c80000" in text and "body paint" in text
    assert "#c8c8c8" in text


def test_both_models_are_image_models(server):
    assert isinstance(OpenRouterImage("m", key="k", url=server.url), ImageModel)
    assert isinstance(HttpImage(server.url), ImageModel)


def test_unknown_size_and_aspect_are_errors():
    model = HttpImage("http://127.0.0.1:1")
    with pytest.raises(ValueError):
        model.generate("x", aspect="wide", size="1K")
    with pytest.raises(ValueError):
        model.generate("x", aspect="1:1", size="huge")

