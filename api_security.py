"""Single-worker API admission, permissions and response protection."""

import hashlib
import logging
import math
import os
import re
import secrets
import threading
import time
from enum import IntEnum
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

from rag_config import Limits

log = logging.getLogger(__name__)
header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Role(IntEnum):
    READER = 1
    EDITOR = 2
    ADMIN = 3


# Route templates, including method, are the unit of authorisation.
POLICY = {
    ("POST", "/query"): (Role.READER, "query"),
    ("POST", "/query/stream"): (Role.READER, "query"),
    ("POST", "/ingest/text"): (Role.EDITOR, "ingest"),
    ("POST", "/ingest/file"): (Role.EDITOR, "ingest"),
    ("GET", "/sources"): (Role.EDITOR, None),
    ("DELETE", "/documents/{doc_id}"): (Role.ADMIN, None),
    ("POST", "/evaluate"): (Role.ADMIN, "evaluate"),
    **{
        ("GET", path): (Role.ADMIN, None)
        for path in ("/metrics", "/traces", "/evals", "/ready")
    },
}
PUBLIC = {
    "/live",
    "/health",
    "/docs",
    "/redoc",
    "/docs/oauth2-redirect",
    "/openapi.json",
}


def origins():
    result = []
    for value in filter(
        None, (s.strip() for s in os.getenv("RAG_CORS_ORIGINS", "").split(","))
    ):
        url = urlsplit(value)
        if (
            url.scheme not in {"https", "http"}
            or not url.hostname
            or url.username is not None
            or url.path
            or url.query
            or url.fragment
            or "*" in value
            or value.endswith(":")
            or any(c.isspace() for c in value)
        ):
            raise ValueError("Invalid RAG_CORS_ORIGINS origin")
        if url.port == 0:
            raise ValueError("Invalid CORS origin port")
        result.append(value)
    return result


class Budget:
    """Fixed one-minute windows. Authenticated identities are the three service roles."""

    def __init__(self, limits, clock=time.monotonic):
        self.limits = limits
        self.clock = clock
        self.lock = threading.Lock()
        self.windows = {}
        self.slots = threading.BoundedSemaphore(limits.concurrent)

    def charge(self, identity, kind):
        now = self.clock()
        with self.lock:
            # At most three roles times five categories plus one anonymous bucket.
            self.windows = {k: v for k, v in self.windows.items() if v[0] > now}
            key = (identity, kind)
            end, count = self.windows.get(key, (now + 60, 0))
            if count >= getattr(self.limits, kind):
                raise HTTPException(
                    429,
                    "Request limit reached",
                    headers={"Retry-After": str(math.ceil(end - now))},
                )
            self.windows[key] = (end, count + 1)


class Access:
    def __init__(self):
        self.keys = {}
        for role in Role:
            value = os.getenv(f"RAG_{role.name}_API_KEY", "")
            if not re.fullmatch("[0-9a-f]{64}", value) or len(set(value)) < 8:
                raise ValueError(f"Set a generated 32-byte hex key for {role.name}")
            self.keys[role] = hashlib.sha256(value.encode()).digest()
        if len(set(self.keys.values())) != len(Role):
            raise ValueError("Service role credentials must be distinct")
        self.limits = Limits.load()
        self.budget = Budget(self.limits)

    def identify(self, value):
        digest = hashlib.sha256((value or "").encode()).digest()
        role = None
        for candidate, expected in self.keys.items():
            if secrets.compare_digest(digest, expected):
                role = candidate
        return role


async def authorize(request: Request, key: str | None = Security(header)):
    access = request.app.state.access
    role = request.state.role
    required, category = POLICY.get(
        (request.method, request.scope["route"].path), (None, None)
    )
    if required is None or role < required:
        raise HTTPException(403, "Insufficient permission")
    if category:
        access.budget.charge(role, category)


async def ai_slot(request: Request):
    if request.app.state.smoke:
        raise HTTPException(503, "AI operations unavailable in smoke-test mode")
    slots = request.app.state.access.budget.slots
    if not slots.acquire(blocking=False):
        raise HTTPException(429, "AI capacity reached", headers={"Retry-After": "1"})
    try:
        yield
    finally:
        slots.release()


class Boundary:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        request_id = secrets.token_hex(12)
        request.state.request_id = request_id
        started = False

        async def respond(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(scope=message)
                headers.update(
                    {
                        "X-Request-ID": request_id,
                        "X-Content-Type-Options": "nosniff",
                        "X-Frame-Options": "DENY",
                        "Referrer-Policy": "no-referrer",
                        "Cache-Control": "no-store",
                    }
                )
                if message["status"] >= 500:
                    log.error(
                        "request_failed request_id=%s status=%d",
                        request_id,
                        message["status"],
                    )
            await send(message)

        try:
            access = request.app.state.access
            public = request.method == "GET" and scope["path"] in PUBLIC
            preflight = (
                request.method == "OPTIONS"
                and "access-control-request-method" in request.headers
                and "origin" in request.headers
            )
            role = access.identify(await header(request))
            if role is None or public or preflight:
                access.budget.charge("shared", "anonymous")
            if not public and not preflight:
                if role is None or len(request.headers.getlist("x-api-key")) != 1:
                    raise HTTPException(
                        401,
                        "Invalid or missing API key",
                        headers={"WWW-Authenticate": "APIKey"},
                    )
                access.budget.charge(role, "general")
                path = scope["path"]
                if re.fullmatch(r"/documents/[^/]+", path):
                    path = "/documents/{doc_id}"
                rule = POLICY.get((request.method, path))
                if rule is None or role < rule[0]:
                    raise HTTPException(403, "Insufficient permission")
                request.state.role = role
            limit = access.limits.body_bytes
            length = request.headers.get("content-length")
            if length and (
                not length.isascii() or not length.isdecimal() or len(length) > 19
            ):
                raise HTTPException(400, "Invalid content length")
            if length and int(length) > limit:
                raise HTTPException(413, "Request body too large")
            if request.method == "POST":
                media = (
                    request.headers.get("content-type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )
                expected = (
                    "multipart/form-data"
                    if scope["path"] == "/ingest/file"
                    else "application/json"
                )
                if media != expected:
                    raise HTTPException(415, "Unsupported request media type")
            # Bound actual bytes before passing any body to multipart or JSON parsers.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > limit:
                    raise HTTPException(413, "Request body too large")
                body.extend(chunk)
                if not message.get("more_body"):
                    break
            pending = True

            async def replay():
                nonlocal pending
                if pending:
                    pending = False
                    return {
                        "type": "http.request",
                        "body": bytes(body),
                        "more_body": False,
                    }
                return await receive()

            await self.app(scope, replay, respond)
        except HTTPException as exc:
            await JSONResponse(
                {"detail": exc.detail}, exc.status_code, headers=exc.headers
            )(scope, receive, respond)
        except Exception:  # noqa: BLE001 - last-resort ASGI response boundary
            if started:
                raise RuntimeError("Response interrupted") from None
            await JSONResponse({"detail": "Request failed"}, 500)(
                scope, receive, respond
            )
