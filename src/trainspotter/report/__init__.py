"""Report renderers: terminal (rich), JSON, and self-contained HTML."""

from __future__ import annotations

from .html_report import render_html
from .json_report import to_json_dict, to_json_str
from .terminal import render_terminal

__all__ = ["render_terminal", "to_json_dict", "to_json_str", "render_html"]
