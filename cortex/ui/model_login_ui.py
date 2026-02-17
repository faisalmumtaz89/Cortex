"""Model login UI helpers for CLI."""

from __future__ import annotations

import getpass
import subprocess
import sys
from typing import Any, Optional

from cortex.cloud.credentials import CloudCredentialStore
from cortex.cloud.types import CloudProvider


def _emit(cli: Any, text: str = "") -> None:
    """Render ANSI-formatted text to the active console/stdout."""
    console = getattr(cli, "console", None)
    if console is not None and hasattr(console, "print"):
        try:
            if text:
                console.print(text)
            else:
                console.print()
            return
        except Exception:
            pass
    if not text:
        print()
        return
    print(text)


def login(*, cli: Any, args: str = "") -> None:
    """Login command router for OpenAI, Anthropic, and HuggingFace."""
    target: Optional[str] = args.strip().lower()

    if not target:
        target = _prompt_login_target(cli=cli)
        if not target:
            return

    if target in {"openai", "anthropic"}:
        _cloud_login(cli=cli, provider=CloudProvider.from_value(target))
        return

    if target in {"huggingface", "hf"}:
        hf_login(cli=cli)
        return

    _emit(cli, f"\033[31m✗\033[0m Unknown login target: \033[93m{target}\033[0m")
    _emit(cli, "\033[2mUse /login openai, /login anthropic, or /login huggingface\033[0m")


def _prompt_login_target(*, cli: Any) -> Optional[str]:
    width = min(cli.get_terminal_width() - 2, 70)

    _emit(cli)
    cli.print_box_header("Login Provider", width)
    cli.print_empty_line(width)
    cli.print_box_line("  \033[93m[1]\033[0m OpenAI API", width)
    cli.print_box_line("  \033[93m[2]\033[0m Anthropic API", width)
    cli.print_box_line("  \033[93m[3]\033[0m HuggingFace", width)
    cli.print_box_line("  \033[93m[4]\033[0m Cancel", width)
    cli.print_empty_line(width)
    cli.print_box_footer(width)

    choice = cli.get_input_with_escape()
    if choice in {None, "4"}:
        return None
    if choice == "1":
        return "openai"
    if choice == "2":
        return "anthropic"
    if choice == "3":
        return "huggingface"

    _emit(cli, "\033[31m✗\033[0m Invalid selection")
    return None


def _cloud_login(*, cli: Any, provider: CloudProvider) -> None:
    """Handle cloud provider API-key login flow."""
    if not getattr(cli.config.cloud, "cloud_enabled", True):
        _emit(cli, "\033[31m✗\033[0m Cloud features are disabled in config.yaml (cloud_enabled=false).")
        return

    if not _ensure_provider_runtime(provider=provider, cli=cli):
        return

    width = min(cli.get_terminal_width() - 2, 70)
    summary = cli.cloud_credentials.get_auth_summary(provider)
    provider_title = _provider_display_name(provider)

    _emit(cli)
    cli.print_box_header(f"{provider_title} API Login", width)
    cli.print_empty_line(width)
    cli.print_box_line(
        f"  Active source: \033[93m{summary['active_source'] or 'none'}\033[0m",
        width,
    )
    cli.print_box_line(
        f"  Env var: \033[93m{summary['env_var']}\033[0m",
        width,
    )
    cli.print_box_line(
        f"  Keychain saved: \033[93m{'yes' if summary['keychain_present'] else 'no'}\033[0m",
        width,
    )
    cli.print_empty_line(width)
    cli.print_box_line("  \033[93m[1]\033[0m Save or replace API key", width)
    cli.print_box_line("  \033[93m[2]\033[0m Remove saved keychain key", width)
    cli.print_box_line("  \033[93m[3]\033[0m Cancel", width)
    cli.print_empty_line(width)
    cli.print_box_footer(width)

    choice = cli.get_input_with_escape()
    if choice in {None, "3"}:
        return

    if choice == "2":
        deleted, message = cli.cloud_credentials.delete_api_key(provider)
        marker = "\033[32m✓\033[0m" if deleted else "\033[31m✗\033[0m"
        _emit(cli, f"\n{marker} {message}")
        if summary["env_present"]:
            _emit(
                cli,
                f"\033[2mEnvironment variable {summary['env_var']} is still set and will remain active.\033[0m"
            )
        return

    if choice != "1":
        _emit(cli, "\033[31m✗\033[0m Invalid selection")
        return

    _emit(cli)
    cli.reserve_bottom_gutter()
    key = getpass.getpass(
        f"\033[96m▶\033[0m Enter {provider_title} API key \033[2m(or press Enter to cancel)\033[0m: "
    ).strip()

    if not key:
        _emit(cli, "\033[2mCancelled\033[0m")
        return

    _emit(cli, f"\n\033[96m⚡\033[0m Validating {provider_title} API key...")
    valid, message = cli.cloud_router.validate_api_key(provider, key)
    if not valid:
        _emit(cli, f"\033[31m✗\033[0m {message}")
        return

    keyring_ready = _ensure_dependency(
        package="keyring>=24.3.0",
        import_name="keyring",
        reason="saving API keys securely in macOS keychain",
        cli=cli,
    )
    if keyring_ready:
        cli.cloud_credentials = CloudCredentialStore()
        cli.cloud_router.credential_store = cli.cloud_credentials

    saved, save_message = cli.cloud_credentials.save_api_key(provider, key)
    if saved:
        _emit(cli, f"\033[32m✓\033[0m {provider_title} API key validated and saved.")
        if summary["env_present"]:
            _emit(
                cli,
                f"\033[2mNote: {summary['env_var']} is set and will take precedence over keychain.\033[0m"
            )
    else:
        _emit(cli, f"\033[33m⚠\033[0m Key validated, but not persisted: {save_message}")
        _emit(cli, f"\033[2mSet {summary['env_var']} to use this key.\033[0m")


def hf_login(*, cli: Any) -> None:
    """Login to HuggingFace for accessing gated models."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub import login as hf_hub_login
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        _emit(cli, "\n\033[31m✗\033[0m huggingface-hub not installed. Install with: pip install huggingface-hub")
        return

    width = min(cli.get_terminal_width() - 2, 70)

    _emit(cli)
    cli.print_box_header("HuggingFace Login", width)
    cli.print_empty_line(width)

    try:
        api = HfApi()
        user_info = api.whoami()
        if user_info:
            username = user_info.get("name", "Unknown")
            cli.print_box_line(
                f"  \033[32m✓\033[0m Already logged in as: \033[93m{username}\033[0m",
                width,
            )
            cli.print_empty_line(width)
            cli.print_box_line("  \033[96mOptions:\033[0m", width)
            cli.print_box_line("  \033[93m[1]\033[0m Login with new token", width)
            cli.print_box_line("  \033[93m[2]\033[0m Logout", width)
            cli.print_box_line("  \033[93m[3]\033[0m Cancel", width)
            cli.print_box_footer(width)

            choice = cli.get_input_with_escape()
            if choice == "1":
                pass
            elif choice == "2":
                from huggingface_hub import logout

                logout()
                _emit(cli, "\n\033[32m✓\033[0m Successfully logged out from HuggingFace")
                return
            else:
                return
    except Exception:
        pass

    _emit(cli)
    cli.print_box_header("HuggingFace Login", width)
    cli.print_empty_line(width)
    cli.print_box_line("  To access gated models, you need a HuggingFace token.", width)
    cli.print_empty_line(width)
    cli.print_box_line("  \033[96m1.\033[0m Get your token from:", width)
    cli.print_box_line("     \033[93mhttps://huggingface.co/settings/tokens\033[0m", width)
    cli.print_empty_line(width)
    cli.print_box_line("  \033[96m2.\033[0m Create a token with \033[93mread\033[0m permissions", width)
    cli.print_empty_line(width)
    cli.print_box_line("  \033[96m3.\033[0m Paste the token below (input hidden)", width)
    cli.print_box_footer(width)

    _emit(cli)
    cli.reserve_bottom_gutter()
    token = getpass.getpass("\033[96m▶\033[0m Enter token \033[2m(or press Enter to cancel)\033[0m: ")

    if not token:
        _emit(cli, "\033[2mCancelled\033[0m")
        return

    _emit(cli, "\n\033[96m⚡\033[0m Authenticating with HuggingFace...")
    try:
        hf_hub_login(token=token, add_to_git_credential=True)

        api = HfApi()
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")

        _emit(cli, f"\033[32m✓\033[0m Successfully logged in as: \033[93m{username}\033[0m")
        _emit(cli, "\033[2m  Token saved for future use\033[0m")
        _emit(cli, "\033[2m  You can now download gated models\033[0m")

    except HfHubHTTPError as e:
        if "Invalid token" in str(e):
            _emit(cli, "\033[31m✗\033[0m Invalid token. Please check your token and try again.")
        else:
            _emit(cli, f"\033[31m✗\033[0m Login failed: {str(e)}")
    except Exception as e:
        _emit(cli, f"\033[31m✗\033[0m Login failed: {str(e)}")


def _provider_display_name(provider: CloudProvider) -> str:
    if provider == CloudProvider.OPENAI:
        return "OpenAI"
    if provider == CloudProvider.ANTHROPIC:
        return "Anthropic"
    return provider.value


def _provider_dependency(provider: CloudProvider) -> tuple[str, str]:
    if provider == CloudProvider.OPENAI:
        return "openai>=1.54.0", "openai"
    if provider == CloudProvider.ANTHROPIC:
        return "anthropic>=0.42.0", "anthropic"
    raise ValueError(f"Unsupported provider: {provider}")


def _ensure_provider_runtime(*, provider: CloudProvider, cli: Any) -> bool:
    package, import_name = _provider_dependency(provider)
    return _ensure_dependency(
        package=package,
        import_name=import_name,
        reason=f"using {_provider_display_name(provider)} cloud models",
        cli=cli,
    )


def _ensure_dependency(
    *,
    package: str,
    import_name: str,
    reason: str,
    cli: Any,
) -> bool:
    try:
        __import__(import_name)
        return True
    except Exception:
        pass

    dependency_msg = (
        f"\n\033[33m⚠\033[0m Missing dependency \033[93m{package}\033[0m "
        f"required for {reason}."
    )
    _emit(cli, dependency_msg)

    install_msg = f"\n\033[96m⬇\033[0m Installing \033[93m{package}\033[0m..."
    _emit(cli, install_msg)
    command = [sys.executable, "-m", "pip", "install", package]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        success_msg = f"\033[32m✓\033[0m Installed \033[93m{package}\033[0m."
        _emit(cli, success_msg)
        return True

    stderr_tail = (result.stderr or "").strip().splitlines()
    error_line = stderr_tail[-1] if stderr_tail else "Unknown installation error."
    error_msg = f"\033[31m✗\033[0m Auto-install failed: {error_line}"
    fallback_msg = f"\033[2mManual fallback: {sys.executable} -m pip install {package}\033[0m"
    _emit(cli, error_msg)
    _emit(cli, fallback_msg)
    return False
