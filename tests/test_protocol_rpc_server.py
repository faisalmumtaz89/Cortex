import io
import json
import time

from cortex.protocol.rpc_server import RpcMethodError, StdioJsonRpcServer


def _run_once(input_lines: str, register_fn):
    stdin = io.StringIO(input_lines)
    stdout = io.StringIO()
    server = StdioJsonRpcServer(stdin=stdin, stdout=stdout)
    register_fn(server)
    server.run_forever()
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_rpc_server_success_response() -> None:
    def _register(server: StdioJsonRpcServer) -> None:
        def _handler(params):
            return {"ok": params.protocol_version == "1.0.0"}

        server.register("app.handshake", _handler)

    responses = _run_once(
        '{"jsonrpc":"2.0","id":1,"method":"app.handshake","params":{"protocol_version":"1.0.0"}}\n',
        _register,
    )
    assert responses[0]["id"] == 1
    assert responses[0]["result"]["ok"] is True


def test_rpc_server_application_error_mapping() -> None:
    def _register(server: StdioJsonRpcServer) -> None:
        def _handler(_params):
            raise RpcMethodError(code=-32000, message="boom")

        server.register("app.handshake", _handler)

    responses = _run_once(
        '{"jsonrpc":"2.0","id":9,"method":"app.handshake","params":{"protocol_version":"1.0.0"}}\n',
        _register,
    )
    assert responses[0]["id"] == 9
    assert responses[0]["error"]["code"] == -32000
    assert responses[0]["error"]["message"] == "boom"


def test_rpc_server_parse_error() -> None:
    responses = _run_once("{bad json\n", lambda _server: None)
    assert responses[0]["error"]["code"] == -32700


def test_rpc_server_dispatches_requests_concurrently() -> None:
    def _register(server: StdioJsonRpcServer) -> None:
        def _handler(_params):
            time.sleep(0.30)
            return {"ok": True}

        server.register("app.handshake", _handler)

    payload = (
        '{"jsonrpc":"2.0","id":1,"method":"app.handshake","params":{"protocol_version":"1.0.0"}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"app.handshake","params":{"protocol_version":"1.0.0"}}\n'
    )
    started = time.perf_counter()
    responses = _run_once(payload, _register)
    elapsed = time.perf_counter() - started

    # Sequential dispatch would be ~0.60s; concurrent dispatch should stay near one sleep interval.
    assert elapsed < 0.55
    response_ids = {item["id"] for item in responses}
    assert response_ids == {1, 2}
