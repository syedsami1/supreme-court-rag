"""Generator module using Groq chat completions API."""

from __future__ import annotations

from typing import Dict, List

import requests


class GroqGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "llama3-8b-8192",
        timeout: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("Groq API key is required.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.max_context_chars = 12000

    def _build_context(self, retrieved_chunks: List[Dict]) -> str:
        parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "unknown")
            page = meta.get("page", "?")
            text = chunk.get("text", "")
            piece = f"[{i}] Source: {filename} (page {page})\n{text}"
            current = "\n\n".join(parts)
            projected = f"{current}\n\n{piece}" if current else piece
            if len(projected) > self.max_context_chars:
                break
            parts.append(piece)
        return "\n\n".join(parts)

    @staticmethod
    def _format_http_error(response: requests.Response) -> str:
        status = response.status_code
        details = response.text
        try:
            payload = response.json()
            err = payload.get("error", {})
            message = err.get("message")
            err_type = err.get("type")
            err_code = err.get("code")
            chunks = [f"status={status}"]
            if message:
                chunks.append(f"message={message}")
            if err_type:
                chunks.append(f"type={err_type}")
            if err_code:
                chunks.append(f"code={err_code}")
            details = ", ".join(chunks)
        except ValueError:
            details = f"status={status}, body={details}"
        return f"Generation error: {details}"

    def generate(self, query: str, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return "I don't know"

        context = self._build_context(retrieved_chunks)
        system_prompt = (
            "You are a legal assistant for Supreme Court judgments.\n"
            "Answer ONLY from the context. If not found, say 'I don't know'.\n"
            "Do not merge facts across different cases unless the question explicitly asks for comparison.\n"
            "When the user asks for constitutional articles, include only Articles of the Constitution and do not include statute sections."
        )
        user_prompt = f"Question: {query}\n\nContext:\n{context}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.url, headers=headers, json=payload, timeout=self.timeout
            )
            if not response.ok:
                return self._format_http_error(response)
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            return f"Generation error: {exc}"
        except (KeyError, IndexError, TypeError) as exc:
            return f"Unexpected Groq response format: {exc}"
