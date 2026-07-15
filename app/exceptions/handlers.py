import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, message: str, error_code: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        logging.getLogger(__name__).warning("api_error", extra={"context": {
            "request_id": getattr(request.state, "request_id", None),
            "endpoint": request.url.path, "status": "error", "error_code": exc.error_code,
        }})
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "status": "error", "message": exc.message, "error_code": exc.error_code},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception("unhandled_api_error", extra={"context": {
            "request_id": getattr(request.state, "request_id", None),
            "endpoint": request.url.path, "error_code": "INTERNAL_ERROR",
        }})
        return JSONResponse(
            status_code=500,
            content={"success": False, "status": "error", "message": "Terjadi kesalahan internal.", "error_code": "INTERNAL_ERROR"},
        )
