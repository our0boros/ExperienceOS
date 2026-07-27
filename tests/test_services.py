from types import SimpleNamespace

from experience_os.config import Config
from experience_os.environment import MockEnvironment
from experience_os.experience_library import ExperienceLibrary
from experience_os.compiler.inductor import HarnessInductor
from experience_os.models import EnvironmentSnapshot, Harness
from experience_os.runtime import Runtime
from experience_os.repository import Repository
from experience_os.retriever import RuntimeRouter
from experience_os.services import ChatService, EmbeddingService, Services
from experience_os.storage import Storage


class FakeChatClient:
    class _Completions:
        @staticmethod
        def create(**kwargs):
            content = '{"ok": true}' if kwargs.get("response_format") else "ok"
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))]
            )

    chat = SimpleNamespace(completions=_Completions())


class FakeStore:
    def __init__(self):
        self.calls = []

    def aggregate_substep_patterns(self, **kwargs):
        self.calls.append(kwargs)
        return []


class FakeEmbedding:
    model_name = "fake-embedding-v1"
    dimension = 2

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [1.0, 0.0]

    def match_intent(self, *args, **kwargs):
        return [], [], []


def test_service_container_exposes_only_current_service_names(tmp_path):
    config = Config(data_dir=tmp_path)
    services = Services.from_config(config, Storage(config))
    assert hasattr(services, "chat")
    assert hasattr(services, "embedding")
    assert not hasattr(services, "llm")
    assert not hasattr(services, "embed")
    services.embedding._storage.close()


def test_service_facades_expose_stable_methods(tmp_path):
    config = Config(data_dir=tmp_path)
    chat = ChatService(config.llm)
    chat._client = FakeChatClient()
    assert chat.complete([]) == "ok"
    assert chat.complete_json([]) == {"ok": True}

    storage = Storage(config)
    embedding = EmbeddingService(config.llm, storage)
    embedding._compute = lambda text: [1.0, 0.0]
    assert embedding.embed("hello") == [1.0, 0.0]
    assert embedding.embed_batch(["hello", "world"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert embedding.model_name
    assert embedding.dimension == 2
    storage.close()


def test_runtime_exposes_shared_stores_and_closes_owned_library(tmp_path):
    config = Config(data_dir=tmp_path)
    library = ExperienceLibrary(tmp_path / "shared.db")
    runtime = Runtime(config, MockEnvironment(), library=library)
    assert runtime.trace_store.library is library
    assert runtime.experience_store.library is library
    assert runtime.artifact_store.library is library
    assert runtime.inductor.experience_store is runtime.experience_store
    runtime.close()
    library.close()


def test_inductor_prefers_injected_experience_store(tmp_path):
    config = Config(data_dir=tmp_path)
    repo = Repository(config)
    store = FakeStore()
    services = SimpleNamespace(chat=FakeChatClient(), embedding=FakeEmbedding())
    inductor = HarnessInductor(config, services, repo, experience_store=store)
    assert inductor._discover_substep_patterns_from_store("exp-1") == {}
    assert store.calls == [{"experiment_id": "exp-1", "min_support": config.induction.min_support}]
    repo.storage.close()


def test_retriever_prefers_injected_embedding_service(tmp_path):
    config = Config(data_dir=tmp_path)
    repo = Repository(config)
    harness = Harness(task_type="lookup", description="find a user")
    repo.add_harness(harness)
    embedding = FakeEmbedding()
    services = SimpleNamespace(embedding=embedding, chat=SimpleNamespace())

    result = RuntimeRouter(repo, services).select(
        "find a user", EnvironmentSnapshot(), task_type="lookup"
    )

    assert result.harness is harness
    assert embedding.calls == ["find a user", harness.retrieval_text()]
    repo.storage.close()


# ── ProviderRegistry ───────────────────────────────────────────────

from experience_os.services import ProviderInfo, ProviderRegistry


def test_provider_registry_has_builtin_providers():
    names = ProviderRegistry.list_names()
    assert "deepinfra" in names
    assert "ollama" in names
    assert "openai" in names
    assert "anthropic" in names
    assert "local" in names
    assert "litellm" in names


def test_provider_registry_get():
    info = ProviderRegistry.get("deepinfra")
    assert info is not None
    assert info.name == "deepinfra"
    assert "deepinfra.com" in info.base_url
    assert info.embedding_dimension == 1024


def test_provider_registry_get_missing():
    assert ProviderRegistry.get("nonexistent") is None


def test_provider_registry_is_registered():
    assert ProviderRegistry.is_registered("ollama") is True
    assert ProviderRegistry.is_registered("fake") is False


def test_provider_registry_register_and_override():
    custom = ProviderInfo(name="test_provider", base_url="http://test:8080/v1",
                          llm_model="test-model")
    ProviderRegistry.register(custom)
    assert ProviderRegistry.is_registered("test_provider") is True
    info = ProviderRegistry.get("test_provider")
    assert info.llm_model == "test-model"


def test_provider_info_to_dict():
    info = ProviderInfo(name="deepinfra", base_url="https://api.deepinfra.com/v1/openai",
                        api_key_env="TOKEN", llm_model="m1",
                        embedding_model="emb1", embedding_dimension=1024,
                        description="test")
    d = info.to_dict()
    assert d["name"] == "deepinfra"
    assert d["llm_model"] == "m1"
    assert d["embedding_dimension"] == 1024


def test_provider_registry_list_all():
    all_providers = ProviderRegistry.list_all()
    names = [p.name for p in all_providers]
    assert "ollama" in names
    assert names == sorted(names)  # sorted by name


def test_provider_registry_resolve_url():
    assert ProviderRegistry.resolve_url("ollama") == "http://localhost:11434/v1"
    assert ProviderRegistry.resolve_url("anthropic") == ""
    assert ProviderRegistry.resolve_url("fake") is None


def test_services_list_providers():
    names = Services.list_providers()
    assert "deepinfra" in names
    assert "ollama" in names
