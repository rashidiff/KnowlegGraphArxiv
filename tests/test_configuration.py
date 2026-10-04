from backend.agents.router import get_llm


def test_documented_llm_environment_variable_names_are_supported(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    llm = get_llm()

    assert str(llm.openai_api_base) == "https://example.test/v1"
    assert llm.model_name == "test-model"
