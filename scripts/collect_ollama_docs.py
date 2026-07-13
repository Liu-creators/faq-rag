from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://docs.ollama.com"
LLMS_TXT_URL = f"{BASE_URL}/llms.txt"
DEFAULT_USER_AGENT = "KnowFlowDocsCollector/0.1 (+learning-project; contact=liu10159287@gmail.com)"

SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".mp4", ".mp3", ".woff",
    ".woff2", ".ttf", ".css", ".js", ".map",
}

SKIP_PATH_PREFIXES = (
    "/images/",
    "/_next/",
    "/assets/",
)


@dataclass
class DocumentRecord:
    document_id: str
    title: str
    source_url: str
    fetched_url: str
    source_type: str
    fetched_at: str
    content_hash: str
    content_file: str
    content_format: str
    http_status: int


def build_session(user_agent: str) -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html, text/markdown, text/plain;q=0.9, */*;q=0.5",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def load_robots(base_url: str, user_agent: str) -> RobotFileParser:
    robots_url = urljoin(base_url, "/robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        parser.read()
        logging.info("已加载 robots.txt：%s", robots_url)
    except Exception as exc:
        logging.warning("无法加载 robots.txt（%s），默认停止。错误：%s", robots_url, exc)
        raise RuntimeError("无法获取 robots.txt；请先手动确认访问权限再爬取。") from exc

    if not parser.can_fetch(user_agent, base_url):
        raise PermissionError(f"robots.txt 禁止此爬虫：{user_agent}")

    return parser


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    normalized = parsed._replace(fragment="", query="")
    return urlunparse(normalized).rstrip("/")


def is_allowed_url(url: str, robots: RobotFileParser, user_agent: str) -> bool:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False

    if parsed.netloc != urlparse(BASE_URL).netloc:
        return False

    path = parsed.path.lower()
    if path.endswith(tuple(SKIP_SUFFIXES)):
        return False

    if any(path.startswith(prefix) for prefix in SKIP_PATH_PREFIXES):
        return False

    return robots.can_fetch(user_agent, url)


def parse_llms_txt(content: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)>\]]+", content)
    return sorted({normalize_url(url) for url in urls})


def extract_internal_links(html: str, page_url: str, robots: RobotFileParser, user_agent: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        absolute_url = normalize_url(urljoin(page_url, href))
        if is_allowed_url(absolute_url, robots, user_agent):
            links.add(absolute_url)

    return sorted(links)


def extract_main_content(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for element in soup.select(
        "script, style, noscript, nav, footer, header, aside, "
        "[role='navigation'], [role='banner'], [role='contentinfo']"
    ):
        element.decompose()

    main = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one("[role='main']")
        or soup.body
        or soup
    )

    title_node = soup.select_one("h1") or soup.title
    title = title_node.get_text(" ", strip=True) if title_node else "未命名"

    markdown = to_markdown(
        str(main),
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )

    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return title, markdown


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{name[:100]}_{digest}.md"


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def make_document_id(source_url: str) -> str:
    digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    return f"ollama_{digest}"


def try_fetch_markdown(session: requests.Session, source_url: str, timeout: int) -> tuple[str, str] | None:
    markdown_url = f"{source_url}.md"

    try:
        response = session.get(markdown_url, timeout=timeout)
    except requests.RequestException:
        return None

    content_type = response.headers.get("Content-Type", "").lower()
    if response.status_code != 200:
        return None

    text = response.text.strip()
    looks_like_markdown = (
        "markdown" in content_type
        or text.startswith("#")
        or "\n## " in text
        or "\n```" in text
    )

    if not text or not looks_like_markdown:
        return None

    first_heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = first_heading.group(1).strip() if first_heading else source_url.rsplit("/", 1)[-1]
    return title, text


def fetch_document(
    session: requests.Session,
    source_url: str,
    timeout: int,
) -> tuple[str, str, str, int] | None:
    markdown_result = try_fetch_markdown(session, source_url, timeout)
    if markdown_result:
        title, content = markdown_result
        return title, content, "markdown", 200

    try:
        response = session.get(source_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.warning("抓取失败：%s | %s", source_url, exc)
        return None

    title, content = extract_main_content(response.text)
    if len(content) < 80:
        logging.warning("内容过短，已跳过：%s", source_url)
        return None

    return title, content, "html_to_markdown", response.status_code


def write_record(record: DocumentRecord, output_dir: Path, manifest_path: Path) -> None:
    content_path = output_dir / record.content_file
    content_path.write_text(
        f"---\n"
        f"document_id: {record.document_id}\n"
        f"title: {json.dumps(record.title, ensure_ascii=False)}\n"
        f"source_url: {record.source_url}\n"
        f"fetched_at: {record.fetched_at}\n"
        f"content_hash: {record.content_hash}\n"
        f"content_format: {record.content_format}\n"
        f"---\n\n",
        encoding="utf-8",
    )

    with content_path.open("a", encoding="utf-8") as file:
        file.write(record._content)

    record_dict = asdict(record)
    record_dict.pop("_content", None)
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record_dict, ensure_ascii=False) + "\n")


def crawl(
    output_dir: Path,
    max_pages: int,
    delay_seconds: float,
    timeout: int,
    user_agent: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.unlink(missing_ok=True)

    session = build_session(user_agent)
    robots = load_robots(BASE_URL, user_agent)

    logging.info("正在获取文档索引：%s", LLMS_TXT_URL)
    response = session.get(LLMS_TXT_URL, timeout=timeout)
    response.raise_for_status()

    seed_urls = parse_llms_txt(response.text)
    if not seed_urls:
        raise RuntimeError("llms.txt 中未找到文档 URL")

    queue = [url for url in seed_urls if is_allowed_url(url, robots, user_agent)]
    visited: set[str] = set()
    saved_count = 0

    logging.info("找到 %d 个允许抓取的种子 URL", len(queue))

    while queue and saved_count < max_pages:
        source_url = normalize_url(queue.pop(0))

        if source_url in visited:
            continue

        visited.add(source_url)
        logging.info("[%d/%d] 正在抓取 %s", saved_count + 1, max_pages, source_url)

        result = fetch_document(session, source_url, timeout)
        if result is None:
            time.sleep(delay_seconds)
            continue

        title, content, content_format, status_code = result
        fetched_at = datetime.now(timezone.utc).isoformat()
        filename = safe_filename_from_url(source_url)

        record = DocumentRecord(
            document_id=make_document_id(source_url),
            title=title,
            source_url=source_url,
            fetched_url=source_url,
            source_type="official_docs",
            fetched_at=fetched_at,
            content_hash=content_hash(content),
            content_file=filename,
            content_format=content_format,
            http_status=status_code,
        )

        record._content = content
        write_record(record, output_dir, manifest_path)
        saved_count += 1

        if content_format == "html_to_markdown":
            try:
                html_response = session.get(source_url, timeout=timeout)
                if html_response.status_code == 200:
                    discovered = extract_internal_links(
                        html_response.text,
                        source_url,
                        robots,
                        user_agent,
                    )
                    for url in discovered:
                        if url not in visited and url not in queue:
                            queue.append(url)
            except requests.RequestException:
                pass

        time.sleep(delay_seconds)

    logging.info("完成。已保存 %d 篇文档到 %s", saved_count, output_dir)
    logging.info("清单文件：%s", manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集 Ollama 官方文档，用于 FAQ + RAG 知识库。"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/ollama_docs",
        help="采集得到的 Markdown 与 manifest.jsonl 的输出目录",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="最多保存的页面数",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="请求间隔（秒）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP 请求超时（秒）",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="爬虫 User-Agent。运行前请替换其中的联系方式。",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()

    try:
        crawl(
            output_dir=Path(args.output_dir),
            max_pages=args.max_pages,
            delay_seconds=args.delay,
            timeout=args.timeout,
            user_agent=args.user_agent,
        )
    except KeyboardInterrupt:
        logging.warning("已由用户中断。")
        sys.exit(130)
    except Exception as exc:
        logging.exception("采集失败：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()