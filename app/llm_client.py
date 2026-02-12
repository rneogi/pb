"""
LLM Client
==========
Wrapper for Anthropic Claude API integration.

Provides RAG-aware response generation with:
    - Context formatting from retrieved chunks
    - System prompt customization
    - Token usage tracking
    - Error handling and fallbacks
"""

import os
from typing import Any, Dict, List, Optional

# Anthropic SDK import (optional - graceful fallback)
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic = None


class ClaudeClient:
    """
    Wrapper for Claude API calls with RAG context support.

    Features:
        - Formats retrieved chunks into context
        - Customizable system prompts
        - Token usage tracking
        - Graceful error handling
    """

    DEFAULT_MODEL = "claude-opus-4-5-20251101"
    DEFAULT_MAX_TOKENS = 1024

    # Default system prompt for PitchBook Observer
    DEFAULT_SYSTEM_PROMPT = """You are the Public PitchBook Observer assistant, a RAG-powered system for analyzing public funding, investment, and company data.

Key principles:
1. ONLY use information from the provided context - do not hallucinate or invent facts
2. Always cite sources using [Source: title] format when making factual claims
3. If information is not in the context, clearly state "Based on the available sources, I don't have information about..."
4. Maintain epistemic humility - this is public data only and may be incomplete
5. For financial claims (funding amounts, valuations), emphasize verification from original sources
6. Be concise but thorough - prioritize accuracy over completeness

Response format:
- Start with a direct answer to the question
- Support claims with citations from the context
- Note any limitations or caveats
- Keep responses focused and well-structured"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (defaults to CLAUDE_MODEL env var or claude-sonnet-4-20250514)
            max_tokens: Maximum response tokens
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL", self.DEFAULT_MODEL)
        self.max_tokens = max_tokens

        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic SDK not installed. Install with: pip install anthropic"
            )

        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self.client = Anthropic(api_key=self.api_key)

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        memory_context: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response using Claude with RAG context.

        Args:
            query: User's question
            context_chunks: Retrieved chunks with metadata
            system_prompt: Optional custom system prompt
            memory_context: Optional previous session context
            max_tokens: Optional override for max tokens

        Returns:
            Dictionary with:
                - text: Generated response
                - model: Model used
                - usage: Token counts
                - citations_used: Number of context chunks
        """
        # Build context from retrieved chunks
        context_text = self._format_context(context_chunks)

        # Build system prompt
        system = self._build_system_prompt(system_prompt, memory_context)

        # Build user message
        user_message = self._build_user_message(query, context_text)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}]
            )

            return {
                "text": response.content[0].text,
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                "citations_used": len(context_chunks),
                "stop_reason": response.stop_reason
            }

        except Exception as e:
            # Return error response
            return {
                "text": f"Error generating response: {str(e)}",
                "model": self.model,
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                "citations_used": len(context_chunks),
                "error": str(e)
            }

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: List of chunk dictionaries with metadata

        Returns:
            Formatted context string for the prompt
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", chunk)

            title = meta.get("title", "Untitled")
            source = meta.get("source_name", "Unknown Source")
            source_kind = meta.get("source_kind", "unknown")
            url = meta.get("canonical_url", "")
            week = meta.get("week", "")
            text = meta.get("text", "")[:800]  # Limit text length

            # Get score (could be rerank_score or score)
            score = chunk.get("rerank_score", chunk.get("score", 0))

            context_parts.append(
                f"[{i}] {title}\n"
                f"Source: {source} ({source_kind}) | Week: {week}\n"
                f"URL: {url}\n"
                f"Relevance: {score:.3f}\n"
                f"Content: {text}\n"
                f"---"
            )

        return "\n\n".join(context_parts)

    def _build_system_prompt(
        self,
        custom_prompt: Optional[str],
        memory_context: Optional[str]
    ) -> str:
        """Build the complete system prompt."""
        base_prompt = custom_prompt or self.DEFAULT_SYSTEM_PROMPT

        if memory_context:
            return f"{base_prompt}\n\n{memory_context}"

        return base_prompt

    def _build_user_message(self, query: str, context_text: str) -> str:
        """Build the user message with context and query."""
        return f"""Based on the following retrieved context, answer the user's question.
Always cite sources using [Source: title] format when making factual claims.
If the context doesn't contain relevant information, say so clearly.

RETRIEVED CONTEXT:
{context_text}

USER QUESTION: {query}

Please provide a comprehensive, well-structured answer with citations from the context above."""

    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        return HAS_ANTHROPIC and self.api_key is not None


class MockClaudeClient:
    """
    Mock client for testing without API calls.
    """

    def __init__(self, *args, **kwargs):
        pass

    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate mock response from context."""
        if not context_chunks:
            text = "No relevant information found in the indexed sources."
        else:
            # Build response from context
            parts = ["Based on the retrieved sources:\n"]
            for i, chunk in enumerate(context_chunks[:3], 1):
                meta = chunk.get("metadata", chunk)
                title = meta.get("title", "Untitled")
                snippet = meta.get("text", "")[:150]
                parts.append(f"\n{i}. **{title}**\n> {snippet}...")

            text = "\n".join(parts)

        return {
            "text": text,
            "model": "mock-model",
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "citations_used": len(context_chunks),
            "mock": True
        }

    def is_available(self) -> bool:
        return True


def get_llm_client(use_mock: bool = False) -> ClaudeClient:
    """
    Factory function to get appropriate LLM client.

    Args:
        use_mock: If True, return mock client for testing

    Returns:
        ClaudeClient or MockClaudeClient instance
    """
    if use_mock:
        return MockClaudeClient()

    try:
        return ClaudeClient()
    except (ImportError, ValueError) as e:
        print(f"Warning: Could not initialize Claude client: {e}")
        print("Falling back to mock client")
        return MockClaudeClient()
