import abc
import json
from typing import Any, Callable, Dict, Optional

import requests


class Target(abc.ABC):
    """Abstract base class for evaluation targets."""

    @abc.abstractmethod
    def evaluate(self, prompt: str) -> Any:
        """
        Evaluates the prompt against the target.

        Args:
            prompt: The full prompt string to send to the target.

        Returns:
            The response from the target. The type depends on the implementation.
        """
        pass


class APITarget(Target):
    """
    A generic API target that sends requests to a specified endpoint.
    """

    def __init__(
        self,
        endpoint: str,
        payload_template: Dict[str, Any],
        prompt_placeholder: str = "{prompt}",
        headers: Optional[Dict[str, str]] = None,
        api_key: Optional[str] = None,
        response_parser: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        """
        Args:
            endpoint: The URL of the API endpoint.
            payload_template: A dictionary representing the JSON payload.
                              It must contain the `prompt_placeholder` which will be replaced by the actual prompt.
            prompt_placeholder: The string in `payload_template` values to be replaced by the prompt.
            headers: Optional headers to include in the request.
            api_key: Optional API key. If provided, it will be added to the headers (default Authorization: Bearer <key>).
            response_parser: A function that takes the JSON response and returns the generated text.
                             If None, the full JSON response is returned.
        """
        self.endpoint = endpoint
        self.payload_template = payload_template
        self.prompt_placeholder = prompt_placeholder
        self.headers = headers or {}
        self.response_parser = response_parser

        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Ensure Content-Type is set
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"

    def _fill_template(self, template: Any, prompt: str) -> Any:
        if isinstance(template, str):
            return template.replace(self.prompt_placeholder, prompt)
        elif isinstance(template, dict):
            return {k: self._fill_template(v, prompt) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._fill_template(item, prompt) for item in template]
        else:
            return template

    def evaluate(self, prompt: str) -> Any:
        payload = self._fill_template(self.payload_template, prompt)

        try:
            response = requests.post(self.endpoint, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if self.response_parser:
                return self.response_parser(data)
            return data
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
