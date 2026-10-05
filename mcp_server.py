from mcp.server.mcpserver import MCPServer
from research_task import KNOWLEDGE_BASE, RELATED

mcp = MCPServer("research-tools")

@mcp.tool()
def search_topic(topic: str) -> str:
    """Get factual information about a topic."""
    key = topic.strip()
    return KNOWLEDGE_BASE.get(key, f"No information about {topic}.")

@mcp.tool()
def get_related(topic: str) -> str:
    """Get related topics for further context."""
    key = topic.strip()
    related = RELATED.get(key, [])
    return ", ".join(related) if related else f"No related topics for {topic}."

if __name__ == "__main__":
    mcp.run(transport="stdio")