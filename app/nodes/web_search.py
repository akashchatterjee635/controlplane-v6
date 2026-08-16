"""Web search node for fallback document retrieval."""

import os
from typing import Any

from langchain_core.documents import Document
from tavily import TavilyClient

from app.state import ControlPlaneState

def web_search_node(state: ControlPlaneState) -> dict[str, Any]:
    """Perform a web search using Tavily.
    
    Appends the search results to the graded_documents list.
    """
    query = state.get("query", "")
    graded_documents = state.get("graded_documents", [])
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "tvly-your-key-here":
        # Fallback if no Tavily key provided
        doc = Document(
            page_content="[Mock Web Search Result] No Tavily API key configured. This is a placeholder web search result.",
            metadata={"source": "web_search", "url": "https://example.com"}
        )
        graded_documents.append(doc)
        return {
            "graded_documents": graded_documents,
            "audit_log": ["[WEB_SEARCH] Mocked web search due to missing TAVILY_API_KEY."],
        }
        
    try:
        client = TavilyClient(api_key=api_key)
        # We can search for the original query. In a more advanced setup, 
        # an LLM could rewrite the query to optimize for web search.
        results = client.search(query=query, search_depth="basic", max_results=3)
        
        docs = []
        for res in results.get("results", []):
            docs.append(
                Document(
                    page_content=res.get("content", ""),
                    metadata={"source": "web_search", "url": res.get("url", "")}
                )
            )
            
        # Append to existing graded documents
        graded_documents.extend(docs)
        
        return {
            "graded_documents": graded_documents,
            "audit_log": [f"[WEB_SEARCH] Successfully retrieved {len(docs)} documents from web search."],
        }
        
    except Exception as e:
        return {
            "audit_log": [f"[WEB_SEARCH] Web search failed: {str(e)}"],
        }
