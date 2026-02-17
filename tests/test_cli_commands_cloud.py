from cortex.ui.cli_commands import CommandHandlers, handle_command


def _make_handlers(calls):
    return CommandHandlers(
        show_help=lambda: calls.append(("help", None)),
        manage_models=lambda args: calls.append(("model", args)),
        download_model=lambda args: calls.append(("download", args)),
        clear_conversation=lambda: calls.append(("clear", None)),
        save_conversation=lambda: calls.append(("save", None)),
        show_status=lambda: calls.append(("status", None)),
        show_gpu_status=lambda: calls.append(("gpu", None)),
        run_benchmark=lambda: calls.append(("benchmark", None)),
        manage_template=lambda args: calls.append(("template", args)),
        run_finetune=lambda: calls.append(("finetune", None)),
        login=lambda args: calls.append(("login", args)),
        show_shortcuts=lambda: calls.append(("shortcuts", None)),
        unknown_command=lambda cmd: calls.append(("unknown", cmd)),
    )


def test_login_command_passes_provider_argument():
    calls = []
    handlers = _make_handlers(calls)
    keep_running = handle_command("/login openai", handlers)
    assert keep_running is True
    assert ("login", "openai") in calls


def test_login_command_without_args_passes_empty_string():
    calls = []
    handlers = _make_handlers(calls)
    keep_running = handle_command("/login", handlers)
    assert keep_running is True
    assert ("login", "") in calls
