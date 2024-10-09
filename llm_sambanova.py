import llm
from llm.default_plugins.openai_models import Chat, Completion

# Hardcoded models for now
def get_sambanova_models():
    return [
        {"id": "Meta-Llama-3.2-1B-Instruct"},
        {"id": "Meta-Llama-3.2-3B-Instruct"},
        {"id": "Meta-Llama-3.1-8B-Instruct"},
        {"id": "Meta-Llama-3.1-8B-Instruct-8k"},
        {"id": "Meta-Llama-3.1-70B-Instruct"},
        {"id": "Meta-Llama-3.1-70B-Instruct-8k"},
        {"id": "Meta-Llama-3.1-405B-Instruct"},
        {"id": "Meta-Llama-3.1-405B-Instruct-8k"},
    ]

class SambaNovaChat(Chat):
    needs_key = "sambanova"
    key_env_var = "SAMBANOVA_KEY"

    def __str__(self):
        return "SambaNova: {}".format(self.model_id)

class SambaNovaCompletion(Completion):
    needs_key = "sambanova"
    key_env_var = "SAMBANOVA_KEY"

    def __str__(self):
        return "SambaNova: {}".format(self.model_id)

@llm.hookimpl
def register_models(register):
    # Only do this if the sambanova key is set
    key = llm.get_key("", "sambanova", "LLM_SAMBANOVA_KEY")
    if not key:
        return

    models = get_sambanova_models()

    for model_definition in models:
        chat_model = SambaNovaChat(
            model_id="sambanova/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            api_base="https://api.sambanova.ai/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(chat_model)

    for model_definition in models:
        completion_model = SambaNovaCompletion(
            model_id="sambanovacompletion/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            api_base="https://api.sambanova.ai/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(completion_model)

class DownloadError(Exception):
    pass
