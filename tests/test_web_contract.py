from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


class ContractHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))


def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def test_required_static_files_and_dom_contract() -> None:
    for name in ("index.html", "styles.css", "api.js", "app.js"):
        assert (WEB / name).is_file(), name

    parser = ContractHTMLParser()
    parser.feed(_read("index.html"))
    required_ids = {
        "login-form",
        "feed-view",
        "feed-tabs",
        "feed-list",
        "profile-dialog",
        "dashboard-view",
        "metric-grid",
        "request-debug-form",
        "operations-view",
        "item-search-form",
        "status-dialog",
        "boost-dialog",
        "operations-body",
    }
    assert required_ids <= parser.ids


def test_exact_frozen_api_paths_are_present() -> None:
    source = _read("api.js")
    required_paths = {
        'const API_BASE = "/api/v1"',
        '"/auth/login"',
        '"/auth/me"',
        '"/auth/logout"',
        '`/feeds/${feedType}',
        '"/events/batch"',
        '"/me/profile"',
        '`/me/events',
        '"/admin/dashboard/overview"',
        '`/admin/requests/${',
        '`/admin/users/${',
        '"/admin/users"',
        '`/admin/items',
        '`/admin/items/${',
        '"/admin/boosts"',
        '"/admin/operations"',
        '"/admin/models"',
    }
    missing = sorted(path for path in required_paths if path not in source)
    assert not missing, missing
    assert 'credentials: "same-origin"' in source


def test_events_use_exposure_linkage_without_identity_override() -> None:
    source = _read("app.js")
    match = re.search(r"api\.sendEvents\(\[\{(?P<body>.*?)\}\]\)", source, re.DOTALL)
    assert match
    event_body = match.group("body")
    for field in ("event_id", "event_type", "request_id", "item_id", "position", "client_timestamp"):
        assert field in event_body
    assert "user_id" not in event_body
    assert 'eventType, "impression"' not in source


def test_admin_ui_is_role_gated_and_server_errors_are_visible() -> None:
    html = _read("index.html")
    app = _read("app.js")
    api_source = _read("api.js")
    assert html.count("admin-only") >= 2
    assert 'state.user?.role === "admin"' in app
    assert "403" in app
    assert "response.status === 401" in api_source
    assert "auth:expired" in api_source
    assert "auth:expired" in app


def test_ui_has_loading_empty_error_cover_and_responsive_states() -> None:
    app = _read("app.js")
    css = _read("styles.css")
    assert "正在生成推荐并记录曝光" in app
    assert "当前没有可展示内容" in app
    assert 'type: "error"' in app
    assert "cover-fallback" in app
    assert 'image.addEventListener("error"' in app
    assert "@media (max-width: 720px)" in css
    assert "grid-template-columns: 1fr" in css


def test_dashboard_and_feed_are_populated_from_api_responses() -> None:
    app = _read("app.js")
    assert "renderDashboard(overviewResult.value)" in app
    assert "response.items || []" in app
    assert "overview.feed_breakdown || []" in app
    assert "overview.top_items || []" in app
    assert "request_id: item._requestId" in app
    assert "position: Number(item.position)" in app


def test_delivery_documents_are_honest_about_verification() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    verification = (ROOT / "docs" / "VERIFICATION.md").read_text(encoding="utf-8")
    demo = (ROOT / "docs" / "DEMO.md").read_text(encoding="utf-8")
    assert "干净 checkout" in readme
    assert "正式演示视频" in readme
    assert "PENDING" in verification
    assert "Not run" in verification
    assert "3–5 Minute" in demo
    assert "MicroLens" in readme
    assert "data/raw/" in readme
