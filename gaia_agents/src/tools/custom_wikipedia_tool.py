import re

import wikipedia
from langchain_core.tools import tool


@tool
def wikipedia_tool(
    title: str, action: str = "summary", section_name: str | None = None, sentences: int = 3
) -> str | list[str]:
    """
    Retrieve information from Wikipedia pages with flexible content extraction.

    This tool provides four main operations for Wikipedia content:
    - Extract summaries of varying lengths
    - Retrieve complete page content including all sections
    - List all section titles to understand page structure
    - Extract specific sections by name with fuzzy matching

    The tool handles section parsing by recognizing Wikipedia's markup format
    (== Section ==, === Subsection ===, etc.) and automatically suggests similar
    sections if exact matches aren't found.

    Args:
        title: Wikipedia page title (supports auto-suggestion for typos)
        action: Operation type - "summary", "full", "sections", or "section"
        section_name: Name of specific section (required when action="section")
        sentences: Number of sentences for summary (default: 3)

    Returns:
        - "summary": Summary text (str)
        - "full": Full page content including all sections (str)
        - "sections": List of all section titles (List[str])
        - "section": Content of matching section or empty string if not found (str)

    Examples:
        wikipedia_tool("Python programming language", "summary")
        wikipedia_tool("Albert Einstein", "full")
        wikipedia_tool("Climate change", "sections")
        wikipedia_tool("Machine learning", "section", section_name="History")
    """

    def parse_sections(content: str) -> list[tuple]:
        """Parse Wikipedia content into sections based on == markers."""
        sections = []

        # Find all section headers with regex
        section_pattern = r"^(={2,})\s*([^=]+?)\s*\1\s*$"
        matches = list(re.finditer(section_pattern, content, re.MULTILINE))

        if not matches:
            return [("Full Content", content.strip())]

        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start_pos = match.end()

            # Find the end position (start of next section or end of content)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)

            section_content = content[start_pos:end_pos].strip()
            sections.append((title, section_content))

        return sections

    # Configure wikipedia settings
    wikipedia.set_lang("en")
    wikipedia.set_rate_limiting(True)

    if action == "summary":
        return wikipedia.summary(title, sentences=sentences, auto_suggest=True)

    elif action == "full":
        page = wikipedia.page(title, auto_suggest=True)
        return page.content

    elif action == "sections":
        page = wikipedia.page(title, auto_suggest=True)
        sections = parse_sections(page.content)
        return [section_title for section_title, _ in sections]

    elif action == "section":
        if not section_name:
            raise ValueError("section_name is required when action='section'")

        page = wikipedia.page(title, auto_suggest=True)
        sections = parse_sections(page.content)

        # Find matching section (fuzzy match)
        for section_title, section_content in sections:
            if section_name.lower() in section_title.lower():
                return section_content

        # If no match found, return empty string
        return "No matching section found. Available sections: " + ", ".join(
            [section_title for section_title, _ in sections]
        )

    else:
        raise ValueError(f"Invalid action: {action}. Must be one of: summary, full, sections, section")
