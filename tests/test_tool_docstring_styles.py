"""Tests for v1.9.0 docstring style support in tool schema generation.

Google style was already supported pre-v1.9.0; NumPy and Sphinx are new.
Detection order is Google → NumPy → Sphinx (first non-empty wins).
"""

from __future__ import annotations

from fastaiagent.tool.function import FunctionTool, _parse_param_descriptions

# ---------------------------------------------------------------------------
# Google style — historical behaviour preserved
# ---------------------------------------------------------------------------


def test_google_style_preserved() -> None:
    def google(x: int, y: str) -> None:
        """Greet someone.

        Args:
            x: The integer value with extra
                description on a continuation line.
            y: The string value.

        Returns:
            Nothing.
        """

    desc = _parse_param_descriptions(google)
    assert desc["x"] == "The integer value with extra description on a continuation line."
    assert desc["y"] == "The string value."


# ---------------------------------------------------------------------------
# NumPy style
# ---------------------------------------------------------------------------


def test_numpy_style_basic() -> None:
    def numpy_fn(x: int, y: str) -> None:
        """Compute something.

        Parameters
        ----------
        x : int
            The x parameter.
        y : str, optional
            The y parameter, with
            multi-line description.

        Returns
        -------
        None
        """

    desc = _parse_param_descriptions(numpy_fn)
    assert desc["x"] == "The x parameter."
    assert desc["y"] == "The y parameter, with multi-line description."


def test_numpy_style_no_type_annotation_in_doc() -> None:
    """NumPy allows ``param`` alone without ``: type``."""

    def numpy_minimal(p: int) -> None:
        """Do a thing.

        Parameters
        ----------
        p
            The parameter.
        """

    desc = _parse_param_descriptions(numpy_minimal)
    assert desc["p"] == "The parameter."


def test_numpy_style_args_section_alias() -> None:
    """NumPy parser also accepts ``Args`` / ``Arguments`` headers."""

    def numpy_alias(z: float) -> None:
        """Numpy with ``Args`` header.

        Args
        ----
        z : float
            The z value.
        """

    desc = _parse_param_descriptions(numpy_alias)
    assert desc["z"] == "The z value."


# ---------------------------------------------------------------------------
# Sphinx / reST style
# ---------------------------------------------------------------------------


def test_sphinx_style_basic() -> None:
    def sphinx_fn(a: int, b: str) -> None:
        """One-line description.

        :param a: The a param, can span
            multiple lines.
        :type a: int
        :param b: The b param.
        :returns: nothing
        """

    desc = _parse_param_descriptions(sphinx_fn)
    assert desc["a"] == "The a param, can span multiple lines."
    assert desc["b"] == "The b param."


def test_sphinx_style_no_returns_section() -> None:
    def sphinx_minimal(p: int) -> None:
        """Process one param.

        :param p: The parameter.
        """

    desc = _parse_param_descriptions(sphinx_minimal)
    assert desc["p"] == "The parameter."


# ---------------------------------------------------------------------------
# Detection order — Google wins when present
# ---------------------------------------------------------------------------


def test_google_wins_over_numpy_when_both_present() -> None:
    def both(x: int) -> None:
        """Both styles present — Google should be preferred.

        Args:
            x: From Google block.

        Parameters
        ----------
        x : int
            From NumPy block.
        """

    desc = _parse_param_descriptions(both)
    assert desc["x"] == "From Google block."


# ---------------------------------------------------------------------------
# No docstring — empty result
# ---------------------------------------------------------------------------


def test_no_docstring_returns_empty() -> None:
    def bare(x: int) -> None:
        pass

    assert _parse_param_descriptions(bare) == {}


def test_docstring_without_param_section_returns_empty() -> None:
    def free_text(x: int) -> None:
        """Just prose. No params block."""

    assert _parse_param_descriptions(free_text) == {}


# ---------------------------------------------------------------------------
# End-to-end through FunctionTool — descriptions reach the JSON schema
# ---------------------------------------------------------------------------


def test_numpy_descriptions_reach_function_tool_schema() -> None:
    def search(query: str, limit: int = 10) -> str:
        """Search for stuff.

        Parameters
        ----------
        query : str
            The search query string.
        limit : int, optional
            How many results to return.

        Returns
        -------
        str
            The first result.
        """
        return ""

    tool = FunctionTool(name="search", fn=search)
    props = tool.parameters["properties"]
    assert props["query"]["description"] == "The search query string."
    assert props["limit"]["description"] == "How many results to return."


def test_sphinx_descriptions_reach_function_tool_schema() -> None:
    def lookup(key: str, default: str = "") -> str:
        """Look up a key.

        :param key: The key to look up.
        :type key: str
        :param default: Returned if the key is absent.
        :type default: str
        :returns: The looked-up value.
        """
        return ""

    tool = FunctionTool(name="lookup", fn=lookup)
    props = tool.parameters["properties"]
    assert props["key"]["description"] == "The key to look up."
    assert props["default"]["description"] == "Returned if the key is absent."
