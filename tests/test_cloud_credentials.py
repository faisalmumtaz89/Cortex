from cortex.cloud.credentials import KEYRING_SERVICE, CloudCredentialStore
from cortex.cloud.types import CloudProvider


class _DummyKeyring:
    def __init__(self):
        self._store = {}

    def set_password(self, service, name, value):
        self._store[(service, name)] = value

    def get_password(self, service, name):
        return self._store.get((service, name))

    def delete_password(self, service, name):
        key = (service, name)
        if key not in self._store:
            raise KeyError(name)
        del self._store[key]


def test_env_var_takes_precedence_over_keychain(monkeypatch):
    store = CloudCredentialStore()
    dummy = _DummyKeyring()
    store._keyring = dummy
    store._keyring_delete_error = KeyError

    dummy.set_password(KEYRING_SERVICE, "openai_api_key", "keychain-openai")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")

    key, source = store.get_api_key_with_source(CloudProvider.OPENAI)
    assert key == "env-openai"
    assert source == "env"


def test_save_read_delete_keychain_key(monkeypatch):
    store = CloudCredentialStore()
    dummy = _DummyKeyring()
    store._keyring = dummy
    store._keyring_delete_error = KeyError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    saved, message = store.save_api_key(CloudProvider.ANTHROPIC, "anthropic-test")
    assert saved is True
    assert "saved" in message.lower()

    key, source = store.get_api_key_with_source(CloudProvider.ANTHROPIC)
    assert key == "anthropic-test"
    assert source == "keychain"

    deleted, delete_message = store.delete_api_key(CloudProvider.ANTHROPIC)
    assert deleted is True
    assert "removed" in delete_message.lower()

    key_after, source_after = store.get_api_key_with_source(CloudProvider.ANTHROPIC)
    assert key_after is None
    assert source_after is None


def test_auth_summary_reports_all_sources(monkeypatch):
    store = CloudCredentialStore()
    dummy = _DummyKeyring()
    store._keyring = dummy
    store._keyring_delete_error = KeyError

    dummy.set_password(KEYRING_SERVICE, "openai_api_key", "keychain-openai")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai")

    summary = store.get_auth_summary(CloudProvider.OPENAI)
    assert summary["active_source"] == "env"
    assert summary["env_present"] is True
    assert summary["keychain_present"] is True
