from typing import Protocol


class ChatBackend(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...

    def health(self) -> dict: ...


class FakeChatBackend:
    def __init__(self, response="INSUFFICIENT_EVIDENCE"):
        self.response = response
        self.calls = []

    def generate(self, *, system_prompt, user_prompt):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response

    def health(self):
        return {"available": True, "backend": "fake", "model_configured": True, "model_loaded": True}
