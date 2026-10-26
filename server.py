from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import agent_core

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"


class Handler(BaseHTTPRequestHandler):
    server_version = "iKUNsight/0.1"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/status":
            return self._json(agent_core.status())
        if path == "/api/model-config":
            return self._json(agent_core.model_settings())
        if path == "/api/model/providers":
            return self._json({"providers": agent_core.MODEL_PROVIDERS})
        if path.startswith("/media/pages/"):
            parts = Path(path.removeprefix("/media/pages/")).parts
            if len(parts) == 2 and re.fullmatch(r"[a-f0-9]{16}", parts[0]) and re.fullmatch(r"page-\d{4}\.png", parts[1]):
                target = agent_core.PAGES / parts[0] / parts[1]
                if target.is_file():
                    return self._file(target)
            return self._json({"error": "页面图像不存在"}, 404)
        if path == "/":
            return self._file(STATIC / "index.html")
        if path in {"/app.css", "/app.js", "/model-settings.css", "/discovery.css"}:
            return self._file(STATIC / path.lstrip("/"))
        return self._json({"error": "未找到"}, 404)

    def _file(self, target: Path):
        if not target.is_file():
            return self._json({"error": "文件不存在"}, 404)
        body = target.read_bytes()
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") or "javascript" in mime else mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/upload":
            return self._upload()
        if path == "/api/ask":
            return self._ask()
        if path == "/api/papers/search":
            return self._paper_search()
        if path == "/api/papers/import":
            return self._paper_import()
        if path == "/api/model-config":
            return self._model_config()
        if path == "/api/model/test":
            return self._model_test()
        return self._json({"error": "未找到"}, 404)

    def _upload(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 35 * 1024 * 1024:
            return self._json({"error": "文件为空或超过 35 MB"}, 413)
        content_type = self.headers.get("Content-Type", "")
        boundary_match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
        if not boundary_match:
            return self._json({"error": "上传格式无效"}, 400)
        raw = self.rfile.read(length)
        boundary = (boundary_match.group(1) or boundary_match.group(2)).encode()
        try:
            parts = raw.split(b"--" + boundary)
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                head, data = part.split(b"\r\n\r\n", 1)
                match = re.search(rb'filename="([^"]+)"', head)
                if match:
                    filename = match.group(1).decode("utf-8", "replace")
                    payload = data.rsplit(b"\r\n", 1)[0]
                    doc = agent_core.save_upload(filename, payload)
                    return self._json({"document": agent_core._doc_json(doc)})
        except Exception as exc:
            return self._json({"error": str(exc)}, 422)
        return self._json({"error": "没有收到 PDF 文件"}, 400)

    def _body_json(self):
        size = int(self.headers.get("Content-Length", "0"))
        if size > 100_000:
            raise ValueError("请求内容过长")
        return json.loads(self.rfile.read(size) or b"{}")

    def _ask(self):
        try:
            data = self._body_json()
            query = str(data.get("query", "")).strip()
            mode = str(data.get("mode", "agent"))
            if not query or len(query) > 2000:
                raise ValueError("请输入不超过 2000 字的问题")
            if mode not in {"agent", "text", "multimodal", "figure", "table"}:
                raise ValueError("未知检索模式")
            return self._json(agent_core.answer(query, mode))
        except ValueError as exc:
            return self._json({"error": str(exc)}, 422)
        except Exception as exc:
            return self._json({"error": f"运行失败：{exc}"}, 500)

    def _paper_search(self):
        try:
            data = self._body_json()
            result = agent_core.search_papers(data.get("query", ""), data.get("sources"), data.get("limit", 12))
            return self._json(result)
        except (ValueError, TypeError) as exc:
            return self._json({"error": str(exc)}, 400)

    def _paper_import(self):
        try:
            data = self._body_json()
            doc = agent_core.import_online_paper(data.get("pdf_url", ""), data.get("title", "paper.pdf"))
            return self._json({"document": agent_core._doc_json(doc)})
        except (ValueError, TypeError) as exc:
            return self._json({"error": str(exc)}, 400)

    def _model_config(self):
        try:
            data = self._body_json()
            settings = agent_core.save_model_settings(data)
            return self._json({"ok": True, **settings, "model_configured": bool(settings["api_key_set"] and settings.get("model"))})
        except (ValueError, TypeError) as exc:
            return self._json({"error": str(exc)}, 400)

    def _model_test(self):
        try:
            data = self._body_json()
            result = agent_core.test_model_connection(data)
            return self._json(result)
        except (ValueError, TypeError) as exc:
            return self._json({"error": str(exc)}, 502)


if __name__ == "__main__":
    host, port = "127.0.0.1", 8766
    print(f"iKUNsight running at http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
