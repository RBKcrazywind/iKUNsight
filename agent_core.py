"""Standalone multimodal scientific-paper RAG prototype.

The prototype deliberately keeps the data path inspectable: every answer is
produced from page anchored evidence, and model use is optional. Without a
configured model it still supports upload, parsing, modality-aware lexical
retrieval, iterative retrieval, and evidence-grounded extractive answers.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlencode, urlparse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency fallback
    PdfReader = None

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
PAGES = DATA / "pages"
for path in (DATA, UPLOADS, PAGES):
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class Evidence:
    id: str
    page: int
    modality: str
    text: str
    score: float = 0.0
    image_url: str = ""
    label: str = ""


@dataclass
class Document:
    id: str
    filename: str
    pages: int
    title: str
    evidence: list[Evidence] = field(default_factory=list)
    page_images: dict[int, str] = field(default_factory=dict)


STATE: dict[str, Any] = {"document": None, "runs": []}
STATE_FILE = DATA / "workspace.json"
MODEL_CONFIG_FILE = DATA / "model_config.json"
MODEL_API_KEY = ""
MODEL_CONFIG: dict[str, Any] = {
    "provider": "custom",
    "base_url": "",
    "api_url": "",
    "protocol": "chat_completions",
    "model": "",
    "vision_model": "",
    "temperature": 0.1,
}

MODEL_PROVIDERS = [
    {"id": "custom", "name": "自定义 OpenAI 兼容服务", "base_url": "", "api_url": "", "protocol": "chat_completions", "model": "", "models": []},
    {"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1", "api_url": "", "protocol": "chat_completions", "model": "gpt-4.1-mini", "models": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o"]},
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com", "api_url": "", "protocol": "chat_completions", "model": "deepseek-chat", "models": ["deepseek-chat", "deepseek-reasoner"]},
    {"id": "ollama", "name": "Ollama（本地）", "base_url": "http://127.0.0.1:11434/v1", "api_url": "", "protocol": "chat_completions", "model": "qwen2.5:7b", "models": []},
]

PAPER_SEARCH_SOURCES = {"arxiv", "openalex", "semantic_scholar"}
PAPER_DOWNLOAD_HOSTS = {
    "arxiv.org", "export.arxiv.org", "api.openalex.org", "semanticscholar.org", "pdfs.semanticscholar.org",
    "www.semanticscholar.org", "api.semanticscholar.org",
}

SEARCH_CONCEPT_ALIASES = {
    "transformer": ["attention is all you need", "self-attention", "sequence transduction"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "multimodal": ["vision-language model", "cross-modal", "multimodal document understanding"],
    "agent": ["tool-using language model", "autonomous agent", "agentic workflow"],
    "computer vision": ["visual recognition", "image understanding", "vision-language"],
    "llm": ["large language model", "foundation model"],
}


def deterministic_search_variants(query: str) -> list[str]:
    """Add a small, inspectable concept expansion layer before source search."""
    variants = [query]
    folded = query.casefold()
    for term, aliases in SEARCH_CONCEPT_ALIASES.items():
        if term in folded:
            variants.extend(aliases)
    return list(dict.fromkeys(variants))[:3]


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 15) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "iKUNsight/0.1 (local research prototype)", **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _clean_abstract(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _openalex_abstract(record: dict[str, Any]) -> str:
    inverted = record.get("abstract_inverted_index") or {}
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions or []:
            words.append((int(position), word))
    return " ".join(word for _, word in sorted(words))


def _search_arxiv(query: str, limit: int) -> list[dict[str, Any]]:
    url = "https://export.arxiv.org/api/query?" + urlencode({
        "search_query": f"all:{query}", "start": 0, "max_results": limit, "sortBy": "relevance",
    })
    request = urllib.request.Request(url, headers={"User-Agent": "iKUNsight/0.1 (local research prototype)"})
    with urllib.request.urlopen(request, timeout=18) as response:
        root = ET.fromstring(response.read())
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("atom:entry", ns):
        links = entry.findall("atom:link", ns)
        pdf = next((item.attrib.get("href", "") for item in links if item.attrib.get("type") == "application/pdf"), "")
        paper_url = entry.findtext("atom:id", "", ns).strip()
        results.append({
            "id": paper_url, "source": "arXiv", "title": _clean_abstract(entry.findtext("atom:title", "", ns)),
            "abstract": _clean_abstract(entry.findtext("atom:summary", "", ns)),
            "year": (entry.findtext("atom:published", "", ns) or "")[:4],
            "authors": [item.findtext("atom:name", "", ns).strip() for item in entry.findall("atom:author", ns)],
            "url": paper_url, "pdf_url": pdf, "doi": "",
        })
    return results


def _search_openalex(query: str, limit: int) -> list[dict[str, Any]]:
    url = "https://api.openalex.org/works?" + urlencode({"search": query, "per-page": limit})
    data = _http_json(url)
    results = []
    for item in data.get("results", [])[:limit]:
        location = item.get("open_access") or {}
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        pdf_url = (primary.get("pdf_url") or "")
        results.append({
            "id": item.get("id", ""), "source": "OpenAlex", "title": _clean_abstract(item.get("title")),
            "abstract": _clean_abstract(_openalex_abstract(item)), "year": str(item.get("publication_year") or ""),
            "authors": [((author.get("author") or {}).get("display_name") or "") for author in item.get("authorships", [])[:6]],
            "url": item.get("doi") or item.get("id") or "", "pdf_url": pdf_url,
            "doi": item.get("doi") or "", "venue": source.get("display_name") or "",
        })
    return results


def _search_semantic_scholar(query: str, limit: int) -> list[dict[str, Any]]:
    fields = "title,abstract,year,authors,url,openAccessPdf,externalIds"
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode({"query": query, "limit": limit, "fields": fields})
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    data = _http_json(url, headers=headers)
    results = []
    for item in data.get("data", [])[:limit]:
        access = item.get("openAccessPdf") or {}
        results.append({
            "id": item.get("paperId", ""), "source": "Semantic Scholar", "title": _clean_abstract(item.get("title")),
            "abstract": _clean_abstract(item.get("abstract")), "year": str(item.get("year") or ""),
            "authors": [author.get("name", "") for author in item.get("authors", [])[:6]],
            "url": item.get("url") or "", "pdf_url": access.get("url") or "", "doi": (item.get("externalIds") or {}).get("DOI", ""),
        })
    return results


def search_papers(query: str, sources: list[str] | None = None, limit: int = 12) -> dict[str, Any]:
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if len(query) < 2 or len(query) > 240:
        raise ValueError("请输入 2 到 240 个字符的研究主题")
    selected = [source for source in (sources or ["arxiv", "openalex", "semantic_scholar"]) if source in PAPER_SEARCH_SOURCES]
    if not selected:
        raise ValueError("至少选择一个论文来源")
    limit = max(1, min(int(limit or 12), 20))
    handlers = {"arxiv": _search_arxiv, "openalex": _search_openalex, "semantic_scholar": _search_semantic_scholar}
    local_variants = deterministic_search_variants(query)
    variants = list(dict.fromkeys([query, *local_variants[1:]]))
    grouped_by_source: dict[str, list[dict[str, Any]]] = {}
    source_status = []
    errors_by_source: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(3, len(selected))) as executor:
        futures = {
            executor.submit(handlers[source], variant, max(4, min(limit, 8))): (source, variant)
            for source in selected for variant in variants
        }
        for future in as_completed(futures):
            source, variant = futures[future]
            try:
                found = future.result()
                for item in found:
                    item["matched_query"] = variant
                grouped_by_source.setdefault(source, []).extend(found)
            except Exception as exc:
                errors_by_source[source] = str(exc)[:180]
    for source in selected:
        found = grouped_by_source.get(source, [])
        source_status.append({"source": source, "ok": bool(found) or source not in errors_by_source,
                              "count": len(found), **({"error": errors_by_source[source]} if source in errors_by_source and not found else {})})
    grouped_results = [grouped_by_source[source] for source in selected if source in grouped_by_source]
    seen: dict[str, dict[str, Any]] = {}
    deduped = []
    max_rows = max((len(group) for group in grouped_results), default=0)
    for row in range(max_rows):
        for group in grouped_results:
            if row >= len(group):
                continue
            item = group[row]
            title_key = re.sub(r"[^a-z0-9]+", " ", item.get("title", "").casefold()).strip()
            doi_key = re.sub(r"^https?://doi\.org/", "", item.get("doi", "").casefold()).strip()
            keys = {key for key in (title_key, doi_key) if key}
            if not keys:
                continue
            duplicate = next((seen[key] for key in keys if key in seen), None)
            if duplicate is not None:
                sources_seen = duplicate.setdefault("matched_sources", [duplicate.get("source", "")])
                if item.get("source") and item["source"] not in sources_seen:
                    sources_seen.append(item["source"])
                queries_seen = duplicate.setdefault("matched_queries", [duplicate.get("matched_query", query)])
                if item.get("matched_query") and item["matched_query"] not in queries_seen:
                    queries_seen.append(item["matched_query"])
                for field in ("abstract", "pdf_url", "doi", "url", "venue"):
                    if len(str(item.get(field) or "")) > len(str(duplicate.get(field) or "")):
                        duplicate[field] = item[field]
                continue
            item["matched_sources"] = [item.get("source", "")]
            item["matched_queries"] = [item.get("matched_query", query)]
            for key in keys:
                seen[key] = item
            item["authors"] = [author for author in item.get("authors", []) if author]
            title_terms = set(_tokens(item.get("title", "")))
            body_terms = set(_tokens(item.get("abstract", "")))
            query_terms = set(_tokens(" ".join(variants)))
            title_hits = query_terms & title_terms
            body_hits = query_terms & body_terms
            score = len(title_hits) * 12 + len(body_hits) * 3
            if any(variant.casefold() in item.get("title", "").casefold() for variant in variants):
                score += 24
            if item.get("pdf_url"):
                score += 2
            try:
                age = max(0, 2026 - int(item.get("year") or 2026))
                score += max(0, 6 - min(age, 6))
            except (TypeError, ValueError):
                pass
            item["relevance_score"] = min(100, score)
            item["relevance_label"] = "高相关" if score >= 35 else "待核验" if score >= 12 else "邻近方向"
            item["relevance_reasons"] = []
            if title_hits:
                item["relevance_reasons"].append("标题命中：" + ", ".join(sorted(title_hits)[:4]))
            if body_hits and not title_hits:
                item["relevance_reasons"].append("摘要命中：" + ", ".join(sorted(body_hits)[:4]))
            if len(item.get("matched_queries", [])) > 1:
                item["relevance_reasons"].append("多个检索表达召回")
            if item.get("pdf_url"):
                item["relevance_reasons"].append("有开放 PDF")
            deduped.append(item)
    deduped.sort(key=lambda item: (-int(item.get("relevance_score") or 0), item.get("year") or "", item.get("title", "").casefold()))
    return {"query": query, "query_variants": variants,
            "results": deduped[:limit], "total": len(deduped), "sources": source_status}


def import_online_paper(url: str, title: str = "paper.pdf") -> Document:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.hostname.casefold() not in PAPER_DOWNLOAD_HOSTS:
        raise ValueError("出于安全原因，只能导入 arXiv、OpenAlex 或 Semantic Scholar 的公开链接")
    request = urllib.request.Request(url, headers={"User-Agent": "iKUNsight/0.1 (local research prototype)", "Accept": "application/pdf"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "").casefold()
            data = response.read(35 * 1024 * 1024 + 1)
    except Exception as exc:
        raise ValueError(f"下载论文失败：{str(exc)[:220]}") from exc
    if len(data) > 35 * 1024 * 1024:
        raise ValueError("论文超过 35 MB，无法导入")
    if not data.startswith(b"%PDF") and "pdf" not in content_type:
        raise ValueError("链接没有返回 PDF 文件，请打开论文页面后下载 PDF")
    filename = title if str(title).lower().endswith(".pdf") else f"{title}.pdf"
    return save_upload(filename, data)


def model_settings() -> dict[str, Any]:
    return {**MODEL_CONFIG, "api_key_set": bool(MODEL_API_KEY)}


def _validate_api_address(label: str, value: str) -> None:
    if not value:
        return
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{label}必须是有效的 http(s) 地址，且不能包含账号、密码或查询参数")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("非本机模型服务请使用 HTTPS")


def save_model_settings(data: dict[str, Any]) -> dict[str, Any]:
    global MODEL_API_KEY, MODEL_CONFIG
    protocol = str(data.get("protocol", "chat_completions")).strip()
    if protocol not in {"chat_completions", "responses"}:
        raise ValueError("请选择 Chat Completions 或 Responses 协议")
    provider = str(data.get("provider", "custom")).strip() or "custom"
    if provider not in {item["id"] for item in MODEL_PROVIDERS}:
        provider = "custom"
    base_url = str(data.get("base_url", "")).strip().rstrip("/")
    api_url = str(data.get("api_url", "")).strip().rstrip("/")
    _validate_api_address("服务地址", base_url)
    _validate_api_address("API 地址", api_url)
    model = str(data.get("model", "")).strip()[:160]
    vision_model = str(data.get("vision_model", "")).strip()[:160]
    temperature = max(0.0, min(2.0, float(data.get("temperature", 0.1))))
    api_key = str(data.get("api_key", "")).strip()
    config = {"provider": provider, "base_url": base_url, "api_url": api_url, "protocol": protocol,
              "model": model, "vision_model": vision_model, "temperature": temperature}
    if data.get("clear_api_key"):
        MODEL_API_KEY = ""
    elif api_key:
        MODEL_API_KEY = api_key
    MODEL_CONFIG = config
    MODEL_CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")
    return model_settings()


def load_model_settings() -> None:
    global MODEL_CONFIG, MODEL_API_KEY
    try:
        saved = json.loads(MODEL_CONFIG_FILE.read_text("utf-8"))
        if isinstance(saved, dict):
            MODEL_CONFIG.update({key: saved[key] for key in MODEL_CONFIG if key in saved})
    except (OSError, json.JSONDecodeError):
        pass
    env_map = {
        "base_url": "PAPER_AGENT_BASE_URL",
        "api_url": "PAPER_AGENT_API_URL",
        "model": "PAPER_AGENT_MODEL",
        "protocol": "PAPER_AGENT_PROTOCOL",
    }
    for config_key, env_key in env_map.items():
        value = os.getenv(env_key, "").strip()
        if value:
            MODEL_CONFIG[config_key] = value
    MODEL_API_KEY = os.getenv("PAPER_AGENT_API_KEY", "").strip()


def model_endpoint(settings: dict[str, Any]) -> str:
    explicit = str(settings.get("api_url", "")).strip().rstrip("/")
    if explicit:
        return explicit
    base = str(settings.get("base_url", "")).strip().rstrip("/")
    suffix = "/responses" if settings.get("protocol") == "responses" else "/chat/completions"
    if base.endswith(("/chat/completions", "/responses")):
        return base
    return base + suffix


def _model_response_text(result: dict[str, Any]) -> str:
    choices = result.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict)).strip()
        if content:
            return str(content).strip()
    if result.get("output_text"):
        return str(result["output_text"]).strip()
    return "".join(str(part.get("text", "")) for item in result.get("output", [])
                   if isinstance(item, dict) for part in item.get("content", [])
                   if isinstance(part, dict)).strip()


def test_model_connection(settings_override: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = {**MODEL_CONFIG, **(settings_override or {})}
    key = str(settings.get("api_key", "")).strip() or MODEL_API_KEY
    model = str(settings.get("model", "")).strip()
    endpoint = model_endpoint(settings)
    if not key or not model or not settings.get("base_url") and not settings.get("api_url"):
        raise ValueError("请填写 API Key、模型名称和 API 地址")
    _validate_api_address("API 地址", endpoint)
    payload = ({"model": model, "instructions": "Reply with OK only.", "input": "Reply with OK only.", "max_output_tokens": 8}
               if settings.get("protocol") == "responses" else
               {"model": model, "messages": [{"role": "user", "content": "Reply with OK only."}], "max_tokens": 8, "temperature": 0})
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    import time
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace").replace(key, "[已隐藏]")
        raise ValueError(f"模型服务返回 HTTP {exc.code}：{detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"连接模型服务失败：{str(exc)[:240]}") from exc
    output = _model_response_text(result)
    if not output:
        raise ValueError("服务已响应，但没有返回可识别的文本；请检查模型与协议是否匹配")
    return {"ok": True, "model": model, "protocol": settings.get("protocol"), "endpoint": endpoint,
            "duration_ms": round((time.perf_counter() - started) * 1000), "output": output[:120]}


load_model_settings()


def _safe_name(name: str) -> str:
    value = re.sub(r"[^\w .()\-\u4e00-\u9fff]", "_", name, flags=re.UNICODE).strip()
    return value[:120] or "paper.pdf"


def _tokens(text: str) -> list[str]:
    parts = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}|[\u4e00-\u9fff]{2,}", text.lower())
    words: list[str] = []
    for part in parts:
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            words.append(part)
            words.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            words.append(part)
    return words


def _query_tokens(text: str) -> list[str]:
    tokens = _tokens(text)
    aliases = {
        "论文": ["paper", "study", "article"], "研究": ["study", "research"],
        "主要": ["main", "key", "primary"], "内容": ["content", "overview"],
        "方法": ["method", "methods", "approach", "methodology"],
        "结果": ["result", "results", "finding", "findings", "performance"],
        "实验": ["experiment", "experiments", "evaluation"],
        "图表": ["figure", "fig", "table", "chart", "plot"],
        "图片": ["figure", "image", "visual"], "图": ["figure", "fig", "chart", "plot"],
        "表": ["table", "tab"], "表格": ["table", "tabular"],
        "贡献": ["contribution", "contributions"], "问题": ["question", "problem"],
        "结论": ["conclusion", "conclusions"], "数据集": ["dataset", "datasets"],
        "准确率": ["accuracy", "precision"], "比较": ["compare", "comparison", "comparative"],
        "性能": ["performance", "metrics"], "模型": ["model", "models"],
        "基线": ["baseline", "baselines"], "限制": ["limitation", "limitations"],
    }
    folded = text.casefold()
    for zh, en in aliases.items():
        if zh in text:
            tokens.extend(en)
    if re.search(r"主要内容|讲了什么|概述", text):
        tokens.extend(["abstract", "overview", "objective", "aim", "purpose"])
    if re.search(r"研究问题|研究目标", text):
        tokens.extend(["research question", "objective", "aim"])
    if "figure" in folded or "fig." in folded:
        tokens.extend(["figure", "fig"])
    if "table" in folded:
        tokens.append("table")
    return tokens


def _render_page(pdf: Path, page: int, out_dir: Path) -> str:
    prefix = out_dir / f"page-{page:04d}"
    target = prefix.with_suffix(".png")
    if target.exists():
        return target.name
    exe = shutil.which("pdftoppm")
    if not exe:
        return ""
    cmd = [exe, "-f", str(page), "-l", str(page), "-png", "-r", "120", str(pdf), str(prefix)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=45)
    except (OSError, subprocess.SubprocessError):
        return ""
    candidates = sorted(out_dir.glob(f"page-{page:04d}-*.png"))
    if candidates and not target.exists():
        candidates[0].replace(target)
    return target.name if target.exists() else ""


def _split_text(text: str, page: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    for start in range(0, len(text), 1100):
        piece = text[start : start + 1250].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _page_modalities(text: str) -> list[tuple[str, str, str]]:
    """Return lightweight table/figure records from page text.

    The page image remains the visual evidence. Captions and dense aligned
    lines give the text retriever useful anchors even when OCR is unavailable.
    """
    found: list[tuple[str, str, str]] = []
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    for line in lines:
        if re.match(r"(?i)^(figure|fig\.?|图)\s*[\dA-Za-z-]+", line):
            found.append(("figure", line, line))
        elif re.match(r"(?i)^(table|tab\.?|表)\s*[\dA-Za-z-]+", line):
            found.append(("table", line, line))
    numeric_lines = [line for line in lines if len(re.findall(r"\d+(?:\.\d+)?", line)) >= 3]
    if len(numeric_lines) >= 2 and not any(kind == "table" for kind, _, _ in found):
        found.append(("table", "Table-like numeric content: " + " | ".join(numeric_lines[:4]), "table"))
    return found


def parse_pdf(path: Path) -> Document:
    if PdfReader is None:
        raise RuntimeError("缺少 pypdf，请先安装：pip install pypdf")
    reader = PdfReader(str(path))
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    page_dir = PAGES / doc_id
    page_dir.mkdir(parents=True, exist_ok=True)
    evidence: list[Evidence] = []
    images: dict[int, str] = {}
    title = path.stem
    for page_no, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if page_no == 1:
            first = next((line.strip() for line in raw.splitlines() if len(line.strip()) > 8), "")
            title = first[:180] or title
        image_name = _render_page(path, page_no, page_dir)
        if image_name:
            images[page_no] = f"/media/pages/{doc_id}/{image_name}"
        for idx, chunk in enumerate(_split_text(raw, page_no), start=1):
            evidence.append(Evidence(f"p{page_no}-t{idx}", page_no, "text", chunk, image_url=images.get(page_no, ""), label="正文"))
        for idx, (kind, text, label) in enumerate(_page_modalities(raw), start=1):
            evidence.append(Evidence(f"p{page_no}-{kind}{idx}", page_no, kind, text, image_url=images.get(page_no, ""), label=label))
        if not raw.strip() and image_name:
            evidence.append(Evidence(f"p{page_no}-visual", page_no, "figure", "该页没有可提取文字，保留页面图像作为视觉证据。", image_url=images[page_no], label="页面图像"))
    return Document(doc_id, path.name, len(reader.pages), title, evidence, images)


def _doc_json(doc: Document | None) -> dict[str, Any] | None:
    if not doc:
        return None
    return {"id": doc.id, "filename": doc.filename, "pages": doc.pages, "title": doc.title,
            "evidence_count": len(doc.evidence), "modalities": {m: sum(e.modality == m for e in doc.evidence) for m in ("text", "figure", "table")}}


def _persist() -> None:
    doc = STATE.get("document")
    payload = {"document": ({"id": doc.id, "filename": doc.filename, "pages": doc.pages,
                             "title": doc.title, "evidence": [asdict(e) for e in doc.evidence],
                             "page_images": {str(k): v for k, v in doc.page_images.items()}} if doc else None),
               "runs": STATE.get("runs", [])[:30]}
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    temp.replace(STATE_FILE)


def _restore() -> None:
    if not STATE_FILE.is_file():
        return
    try:
        saved = json.loads(STATE_FILE.read_text("utf-8"))
        raw = saved.get("document")
        if raw:
            source = UPLOADS / _safe_name(raw["filename"])
            if source.is_file():
                STATE["document"] = Document(raw["id"], raw["filename"], int(raw["pages"]), raw["title"],
                    [Evidence(**item) for item in raw.get("evidence", [])],
                    {int(k): v for k, v in raw.get("page_images", {}).items()})
        STATE["runs"] = saved.get("runs", [])[:30]
    except (OSError, ValueError, KeyError, TypeError):
        return


_restore()


def _retrieve(doc: Document, query: str, mode: str = "agent", limit: int = 8) -> tuple[list[Evidence], dict[str, Any]]:
    terms = _query_tokens(query)
    visual_query = bool(re.search(r"figure|fig|table|chart|plot|image|visual|图|表|曲线|结果|实验", query, re.I))
    if mode == "text": allowed = {"text"}
    elif mode == "multimodal": allowed = {"text", "figure", "table"}
    elif mode == "figure": allowed = {"figure"}
    elif mode == "table": allowed = {"table"}
    else: allowed = {"text", "figure", "table"}
    scored: list[Evidence] = []
    for item in doc.evidence:
        if item.modality not in allowed:
            continue
        text_terms = _tokens(item.text)
        overlap = sum(text_terms.count(term) for term in terms)
        phrase = 2.0 if query.lower() in item.text.lower() else 0.0
        modality_bonus = 2.0 if visual_query and item.modality in {"figure", "table"} else 0.0
        item.score = overlap + phrase + modality_bonus + (0.35 if item.modality != "text" else 0.0)
        if item.score > 0:
            scored.append(item)
    scored.sort(key=lambda x: (-x.score, x.page, x.id))
    chosen: list[Evidence] = []
    pages = set()
    for item in scored:
        if item.page not in pages or len(chosen) < limit:
            chosen.append(item); pages.add(item.page)
        if len(chosen) >= limit:
            break
    return chosen, {"mode": mode, "terms": terms, "candidate_count": len(scored), "visual_query": visual_query}


def _llm_answer(query: str, evidence: list[Evidence], doc: Document) -> str | None:
    settings = MODEL_CONFIG
    key = MODEL_API_KEY
    url = model_endpoint(settings)
    has_images = any(item.image_url for item in evidence)
    selected_model = (settings.get("vision_model") or settings.get("model")) if has_images else settings.get("model")
    model = str(selected_model or "").strip()
    send_images = bool(settings.get("vision_model"))
    if not key or not url or not model:
        return None
    joined = "\n\n".join(f"[Page {e.page} · {e.modality}] {e.text}" for e in evidence)
    prompt = ("You are a scientific paper understanding assistant. Answer only from the supplied evidence. "
              "If evidence is insufficient, say so. Cite page numbers and distinguish text, figure, and table evidence.\n\n"
              f"Paper: {doc.title}\nQuestion: {query}\nEvidence:\n{joined}")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    seen_pages: set[int] = set()
    for item in evidence:
        if not send_images:
            break
        if item.page in seen_pages or not item.image_url or len(seen_pages) >= 3:
            continue
        image_path = PAGES / doc.id / Path(item.image_url).name
        if not image_path.is_file():
            continue
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"
        if settings.get("protocol") == "responses":
            content.append({"type": "input_image", "image_url": data_url})
        else:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        seen_pages.add(item.page)
    if settings.get("protocol") == "responses":
        body = {"model": model, "instructions": "Answer only from supplied scientific-paper evidence. Cite page numbers and distinguish text, figure and table evidence. State when evidence is insufficient.",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}, *content[1:]]}],
                "temperature": settings.get("temperature", 0.1)}
    else:
        body = {"model": model, "temperature": settings.get("temperature", 0.1), "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=75) as response:
            result = json.loads(response.read().decode("utf-8"))
        return _model_response_text(result)
    except (OSError, KeyError, IndexError, json.JSONDecodeError):
        return None


def answer(query: str, mode: str = "agent") -> dict[str, Any]:
    doc: Document | None = STATE.get("document")
    if not doc:
        raise ValueError("请先上传一篇科学论文 PDF")
    steps = []
    if mode == "agent":
        first_mode = "multimodal" if re.search(r"图|表|figure|table|chart|结果|实验", query, re.I) else "text"
        first, meta = _retrieve(doc, query, first_mode, 6)
        steps.append({"agent": "PlannerAgent", "action": f"识别问题并选择 {first_mode} 检索"})
        steps.append({"agent": "RetrievalAgent", "action": f"检索 {len(first)} 条候选证据"})
        if len(first) < 2 or (meta["visual_query"] and not any(e.modality != "text" for e in first)):
            extra, extra_meta = _retrieve(doc, query, "multimodal", 8)
            by_id = {e.id: e for e in first}; by_id.update({e.id: e for e in extra}); first = list(by_id.values())[:8]
            steps.append({"agent": "RoutingAgent", "action": "证据不足，补充跨模态检索"})
            meta = extra_meta
        evidence = first
    else:
        evidence, meta = _retrieve(doc, query, mode, 8)
        steps.append({"agent": "RetrievalAgent", "action": f"执行 {mode} 检索"})
    llm = _llm_answer(query, evidence, doc)
    if llm:
        text = llm
        steps.append({"agent": "AnswerAgent", "action": "基于证据生成回答"})
    elif evidence:
        excerpts = [f"第 {e.page} 页（{e.label or e.modality}）：{e.text[:420]}" for e in evidence[:4]]
        text = "当前未配置外部多模态模型，先返回可核验的证据摘要：\n\n" + "\n\n".join(excerpts)
        steps.append({"agent": "EvidenceAgent", "action": "使用本地抽取式回答，未调用外部模型"})
    else:
        text = "没有检索到足够证据。可以尝试更具体的关键词，或切换到多模态检索。"
        steps.append({"agent": "CriticAgent", "action": "证据不足，要求用户缩小问题"})
    run = {"query": query, "mode": mode, "steps": steps, "retrieval": meta, "evidence": [asdict(e) for e in evidence], "answer": text}
    STATE["runs"].insert(0, run); STATE["runs"] = STATE["runs"][:30]
    _persist()
    return run


def status() -> dict[str, Any]:
    return {"document": _doc_json(STATE.get("document")), "runs": STATE["runs"][:10],
            "model_configured": bool(MODEL_API_KEY and MODEL_CONFIG.get("model")),
            "capabilities": ["PDF 文本提取", "页面图像证据", "图注/表格线索抽取", "模态路由", "Agent 迭代检索", "证据页码追踪"]}


def save_upload(filename: str, data: bytes) -> Document:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("目前只支持 PDF 文件")
    safe = _safe_name(filename)
    target = UPLOADS / safe
    target.write_bytes(data)
    doc = parse_pdf(target)
    STATE["document"] = doc; STATE["runs"] = []
    _persist()
    return doc
