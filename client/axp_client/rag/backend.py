from typing import Protocol


class ChatBackend(Protocol):
    def ensure_loaded(self) -> None: ...

    def retry_load(self) -> None: ...

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...

    def health(self) -> dict: ...

    def count_tokens(self, text: str) -> int: ...

    def context_window(self) -> int: ...


class FakeChatBackend:
    def __init__(self, response="INSUFFICIENT_EVIDENCE"):
        self.response = response
        self.calls = []

    def generate(self, *, system_prompt, user_prompt):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response

    def ensure_loaded(self):
        return None

    def retry_load(self):
        return None

    def health(self):
        return {"available": True, "backend": "fake", "model_configured": True,
                "model_loaded": True, "model_state": "loaded"}

    def count_tokens(self, text):
        return len(text.split())

    def context_window(self):
        return 8192
