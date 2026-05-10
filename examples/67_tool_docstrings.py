"""Example 67: Tool descriptions auto-extracted from any docstring style.

As of v1.9.0, ``FunctionTool`` extracts parameter descriptions from
Google, NumPy, and Sphinx (reST) docstring conventions. Detection order
is Google → NumPy → Sphinx; first style with a match wins.

Runnable as pytest (no API keys, no network):
    pytest examples/67_tool_docstrings.py -v
"""

from fastaiagent.tool.function import FunctionTool


def test_numpy_style_tool() -> None:
    def search(query: str, limit: int = 10) -> str:
        """Search the corpus.

        Parameters
        ----------
        query : str
            The search query string.
        limit : int, optional
            Maximum number of results.
        """
        return ""

    tool = FunctionTool(name="search", fn=search)
    assert tool.parameters["properties"]["query"]["description"] == (
        "The search query string."
    )
    assert tool.parameters["properties"]["limit"]["description"] == (
        "Maximum number of results."
    )


def test_sphinx_style_tool() -> None:
    def lookup(key: str, default: str = "") -> str:
        """Look up a key in storage.

        :param key: The key to look up.
        :type key: str
        :param default: Returned if the key is absent.
        """
        return ""

    tool = FunctionTool(name="lookup", fn=lookup)
    assert tool.parameters["properties"]["key"]["description"] == (
        "The key to look up."
    )
    assert tool.parameters["properties"]["default"]["description"] == (
        "Returned if the key is absent."
    )


def test_google_style_still_works() -> None:
    def greet(name: str, formal: bool = False) -> str:
        """Greet someone.

        Args:
            name: The person's name.
            formal: If True, use a formal greeting.
        """
        return ""

    tool = FunctionTool(name="greet", fn=greet)
    assert tool.parameters["properties"]["name"]["description"] == (
        "The person's name."
    )
    assert tool.parameters["properties"]["formal"]["description"] == (
        "If True, use a formal greeting."
    )
