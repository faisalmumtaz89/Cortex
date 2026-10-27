"""`cortex login PROVIDER [URL]`: save a provider's key without the
interactive app, for machines that only run `cortex -p` (CI, git hooks).

The key is read from stdin, or typed at a hidden prompt in a terminal; it
is never taken from the command line, where shell history and process
listings would keep it. It is saved where /login saves it.
"""

from __future__ import annotations

import getpass
import sys
from typing import List

from cortex.app.model_service import LOGIN_URL_ARGUMENT, ModelService
from cortex.cloud.types import CloudProvider

_PROVIDERS = ", ".join(sorted(provider.value for provider in CloudProvider))


def _fail(message: str) -> int:
    sys.stderr.write(f"error: {message}\n")
    return 2


def _read_key(provider: str) -> str:
    if sys.stdin.isatty():
        return getpass.getpass(f"API key for {provider} (not shown): ").strip()
    return sys.stdin.read().strip()


def run_login(argv: List[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        sys.stderr.write(
            "usage: cortex login PROVIDER [URL]\n\n"
            f"Saves PROVIDER's API key ({_PROVIDERS}) where /login saves it. The key\n"
            "is read from stdin, or asked for in a terminal. azure takes its endpoint\n"
            "and openai-compatible its base URL, unless one is saved already.\n"
        )
        return 0 if argv else 2
    name = argv[0].strip().lower()
    try:
        provider = CloudProvider.from_value(name)
    except ValueError:
        return _fail(f"unknown provider {name}. Use one of: {_PROVIDERS}")
    url_argument = LOGIN_URL_ARGUMENT.get(provider)
    if len(argv) > 1 and not url_argument:
        return _fail(
            f"{provider.value} takes no URL. cortex login reads the key from stdin, or asks "
            "for it in a terminal: never put it on the command line"
        )
    if len(argv) > 2:
        return _fail(
            f"cortex login {provider.value} takes only its {url_argument} after the provider; "
            "it reads the key from stdin, or asks for it in a terminal"
        )

    service = _model_service()
    url = argv[1] if len(argv) == 2 else ""
    if url_argument and not url and not service.endpoint_url(provider):
        what = "endpoint" if provider == CloudProvider.AZURE else "base URL"
        return _fail(
            f"{provider.value} also needs its {what}: "
            f"cortex login {provider.value} {url_argument}"
        )

    key = _read_key(provider.value)
    if not key:
        return _fail(
            f'no key on stdin. Pipe it in (printf %s "$KEY" | cortex login {provider.value}), '
            f"or run cortex login {provider.value} in a terminal to type it"
        )
    if any(ch.isspace() for ch in key):
        return _fail("the key has spaces or line breaks in it: send only the key")

    messages = []
    if url:
        saved_url = service.save_endpoint_url(provider, url)
        if not saved_url["ok"]:
            return _fail(str(saved_url["message"]))
        messages.append(str(saved_url["message"]))
    saved = service.auth_save_key(provider, key)
    if not saved["ok"]:
        return _fail(str(saved["message"]))
    messages.insert(0, str(saved["message"]))
    sys.stdout.write(" ".join(messages) + "\n")
    return 0


def _model_service() -> ModelService:
    from cortex.cloud.catalog import CloudModelCatalog
    from cortex.cloud.credentials import CloudCredentialStore
    from cortex.cloud.router import CloudRouter
    from cortex.config import Config

    config = Config()
    store = CloudCredentialStore()
    return ModelService(
        config=config,
        cloud_router=CloudRouter(config, store),
        credential_store=store,
        cloud_catalog=CloudModelCatalog(),
    )
