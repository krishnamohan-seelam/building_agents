"""Render Tree of Thought agent runs into a self-contained HTML report."""

import html
from datetime import datetime
from pathlib import Path
from string import Template
from typing import List

TEMPLATE_PATH = Path(__file__).with_name("templates") / "report_template.html"


def _score_class(score) -> str:
    if score is None:
        return ""
    if score >= 8.0:
        return "score-high"
    if score >= 5.0:
        return "score-mid"
    return "score-low"


def _thought_rows(all_thoughts: List[dict]) -> str:
    ordered = sorted(
        all_thoughts,
        key=lambda t: (t.get("score") is None, -(t.get("score") or 0)),
    )
    rows = []
    for t in ordered:
        score = t.get("score")
        score_str = f"{score:.1f}" if score is not None else "-"
        status = t.get("status", "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(t['id'])}</td>"
            f"<td class=\"status-{html.escape(status)}\">{html.escape(status)}</td>"
            f"<td class=\"{_score_class(score)}\">{score_str}</td>"
            f"<td>{html.escape(t.get('content', ''))}</td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="4">No thoughts recorded.</td></tr>'


def generate_html_report(
    *,
    problem: str,
    final_answer: str,
    thread_id: str,
    all_thoughts: List[dict],
    depth: int,
    latency_seconds: float,
    output_path: str,
) -> Path:
    """Render a ToT run into an HTML report and save it to `output_path`.

    `output_path` may be a directory (a timestamped filename is generated inside
    it) or a file path (`.html` is appended if missing).
    """
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    rendered = template.substitute(
        title="Tree of Thought Report",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        thread_id=html.escape(thread_id),
        problem=html.escape(problem),
        final_answer=html.escape(final_answer),
        total_thoughts=str(len(all_thoughts)),
        depth=str(depth),
        latency_seconds=f"{latency_seconds:.2f}",
        thought_rows=_thought_rows(all_thoughts),
    )

    path = Path(output_path).expanduser()
    if path.is_dir():
        path = path / f"tot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    elif path.suffix.lower() != ".html":
        path = path.with_suffix(".html")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path
