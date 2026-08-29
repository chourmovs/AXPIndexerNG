from .llama_cpp_backend import LlamaCppBackend


class ChatBackendConfigurationError(ValueError):
    pass


def create_chat_backend(settings):
    backend = settings.get("chat_backend")
    if backend == "llama_cpp":
        return LlamaCppBackend(settings["chat_model_path"])
    raise ChatBackendConfigurationError(f"unsupported chat backend: {backend!r}")
