"""Muse Glimmer client for OpenAI-compatible local or hosted servers."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class GlimmerError(RuntimeError):
    """Raised when the Muse Glimmer endpoint cannot complete a request."""


@dataclass(frozen=True)
class ToolDecision:
    name: str
    arguments: dict[str, Any]
    raw_response: dict[str, Any]


class GlimmerClient:
    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: int = 180) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @staticmethod
    def _image_data_url(path: str) -> str:
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise GlimmerError(f"Glimmer HTTP {error.code}: {body}") from error
        except URLError as error:
            raise GlimmerError(f"Cannot reach Glimmer endpoint: {error}") from error

    def select_tool(self, goal: str, media_path: str, tool_schema: dict[str, Any]) -> ToolDecision:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Use SAM 3 for precise segmentation. Never invent masks or measurements."},
                {"role": "user", "content": [
                    {"type": "text", "text": goal},
                    {"type": "image_url", "image_url": {"url": self._image_data_url(media_path)}},
                ]},
            ],
            "tools": [tool_schema],
            "tool_choice": "auto",
            "temperature": 0,
        }
        response = self._post(payload)
        try:
            tool_call = response["choices"][0]["message"]["tool_calls"][0]
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return ToolDecision(tool_call["function"]["name"], arguments, response)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise GlimmerError("Response did not contain a valid tool call") from error

    def summarize(self, goal: str, tool_name: str, tool_arguments: dict[str, Any], tool_result: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Summarize only verified values in the tool result."},
                {"role": "user", "content": goal},
                {"role": "assistant", "tool_calls": [{
                    "id": "sam3_call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_arguments)},
                }]},
                {"role": "tool", "tool_call_id": "sam3_call_1", "content": json.dumps(tool_result)},
            ],
            "temperature": 0,
        }
        response = self._post(payload)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise GlimmerError("Response did not contain a final summary") from error
