"""Safe errors shared by the Muse client and OpenAI-compatible API."""

from __future__ import annotations


class ProxyError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "muse_error",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.param = param

    def as_dict(self) -> dict:
        error_type = {
            401: "authentication_error",
            403: "permission_error",
            429: "rate_limit_error",
        }.get(
            self.status_code, "server_error" if self.status_code >= 500 else "invalid_request_error"
        )
        return {
            "message": self.message,
            "type": error_type,
            "param": self.param,
            "code": self.code,
        }
