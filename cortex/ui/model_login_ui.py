"""Model login UI helpers for CLI."""

from __future__ import annotations

import getpass
from typing import Any


def hf_login(*, cli: Any) -> None:
    """Login to HuggingFace for accessing gated models."""
    try:
        from huggingface_hub import login, HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print("\n\033[31m✗\033[0m huggingface-hub not installed. Install with: pip install huggingface-hub")
        return

    width = min(cli.get_terminal_width() - 2, 70)

    print()
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

            choice = cli.get_input_with_escape("Select option (1-3)")
            if choice == "1":
                pass
            elif choice == "2":
                from huggingface_hub import logout

                logout()
                print("\n\033[32m✓\033[0m Successfully logged out from HuggingFace")
                return
            else:
                return
    except Exception:
        pass

    print()
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

    print()
    token = getpass.getpass("\033[96m▶\033[0m Enter token \033[2m(or press Enter to cancel)\033[0m: ")

    if not token:
        print("\033[2mCancelled\033[0m")
        return

    print("\n\033[96m⚡\033[0m Authenticating with HuggingFace...")
    try:
        login(token=token, add_to_git_credential=True)

        api = HfApi()
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")

        print(f"\033[32m✓\033[0m Successfully logged in as: \033[93m{username}\033[0m")
        print("\033[2m  Token saved for future use\033[0m")
        print("\033[2m  You can now download gated models\033[0m")

    except HfHubHTTPError as e:
        if "Invalid token" in str(e):
            print("\033[31m✗\033[0m Invalid token. Please check your token and try again.")
        else:
            print(f"\033[31m✗\033[0m Login failed: {str(e)}")
    except Exception as e:
        print(f"\033[31m✗\033[0m Login failed: {str(e)}")
