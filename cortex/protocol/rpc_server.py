"""Line-delimited JSON-RPC 2.0 server for Cortex worker mode."""

from __future__ import annotations

import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import TextIOBase
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ValidationError

from cortex.protocol.schema import parse_method_params
from cortex.protocol.types import RpcError, RpcErrorResponse, RpcRequest, RpcSuccessResponse

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
RpcHandler = Callable[[BaseModel], Any]


class RpcMethodError(RuntimeError):
    """Application error mapped to JSON-RPC response codes."""

    def __init__(self, *, code: int, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass
class RpcMethod:
    """Registered RPC method handler."""

    name: str
    handler: RpcHandler


class StdioJsonRpcServer:
    """Synchronous JSON-RPC server over stdio with strict request validation."""

    def __init__(
        self,
        *,
        stdin: Optional[TextIOBase] = None,
        stdout: Optional[TextIOBase] = None,
        max_workers: int = 8,
    ) -> None:
        self._methods: Dict[str, RpcMethod] = {}
        self._running = False
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._max_workers = max(1, int(max_workers))
        self._write_lock = threading.Lock()

    def register(self, name: str, handler: RpcHandler) -> None:
        if not name.strip():
            raise ValueError("RPC method name cannot be empty")
        self._methods[name] = RpcMethod(name=name, handler=handler)

    def send_raw(self, payload: JsonDict) -> None:
        wire = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        with self._write_lock:
            self._stdout.write(wire)
            self._stdout.write("\n")
            self._stdout.flush()

    def send_success(self, request_id: Any, result: Dict[str, Any]) -> None:
        response = RpcSuccessResponse(id=request_id, result=result)
        self.send_raw(response.model_dump())

    def send_error(
        self,
        *,
        code: int,
        message: str,
        request_id: Optional[Any] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        response = RpcErrorResponse(
            id=request_id,
            error=RpcError(code=code, message=message, data=data),
        )
        self.send_raw(response.model_dump())

    def _dispatch(self, request: RpcRequest) -> None:
        method = self._methods.get(request.method)
        if method is None:
            self.send_error(code=-32601, message=f"Method not found: {request.method}", request_id=request.id)
            return

        try:
            params_model = parse_method_params(request.method, request.params)
        except ValidationError as exc:
            self.send_error(
                code=-32602,
                message="Invalid params",
                request_id=request.id,
                data={"errors": exc.errors()},
            )
            return
        except ValueError as exc:
            self.send_error(code=-32601, message=str(exc), request_id=request.id)
            return

        try:
            result = method.handler(params_model)
            if isinstance(result, BaseModel):
                payload = result.model_dump()
            elif isinstance(result, dict):
                payload = result
            elif result is None:
                payload = {}
            else:
                payload = {"value": result}
            self.send_success(request.id, payload)
        except RpcMethodError as exc:
            self.send_error(
                code=exc.code,
                message=exc.message,
                request_id=request.id,
                data=exc.data,
            )
        except Exception as exc:  # pragma: no cover - integration failure path
            logger.exception("RPC handler failed for method=%s", request.method)
            self.send_error(
                code=-32603,
                message="Internal error",
                request_id=request.id,
                data={"method": request.method, "error": str(exc)},
            )

    def run_forever(self) -> None:
        self._running = True
        with ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="cortex-rpc") as executor:
            while self._running:
                line = self._stdin.readline()
                if line == "":
                    break

                raw = line.strip()
                if not raw:
                    continue

                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    self.send_error(code=-32700, message="Parse error", data={"error": str(exc)})
                    continue

                try:
                    request = RpcRequest.model_validate(payload)
                except ValidationError as exc:
                    self.send_error(code=-32600, message="Invalid Request", data={"errors": exc.errors()})
                    continue

                executor.submit(self._dispatch, request)

    def shutdown(self) -> None:
        self._running = False
