"""
click_select_widget
--------------------
A small self-contained Streamlit component that renders a real native
browser <select> dropdown. Unlike st.selectbox (which is a searchable
combobox you can type/filter into), a native <select> has no text
field at all — you can only click it open and click an option. This
guarantees the cursor is a plain pointer, never a text/edit cursor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Any

import streamlit.components.v1 as components

_frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
    "click_select_widget", path=str(_frontend_dir)
)


def click_select(
    label: str,
    options: Sequence[Any],
    format_func: Callable[[Any], str] | None = None,
    index: int = 0,
    key: str | None = None,
):
    """
    A click-only dropdown (native HTML <select>).

    Parameters
    ----------
    label : str
        Label shown above the dropdown.
    options : sequence
        The list of selectable option values.
    format_func : callable, optional
        Maps each option to its displayed label text.
    index : int
        Default selected index.
    key : str, optional
        Unique widget key.

    Returns
    -------
    The selected option (same type/value as in `options`).
    """

    import streamlit as st

    st.markdown(f"**{label}**")

    options = list(options)
    labels = [format_func(o) for o in options] if format_func else None

    result = _component_func(
        options=options,
        labels=labels,
        index=index,
        key=key,
        default=options[index] if options else None,
    )

    return result if result is not None else (options[index] if options else None)
