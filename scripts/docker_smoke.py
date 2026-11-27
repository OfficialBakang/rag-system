"""Verify the built runtime with Docker CLI, disposable storage and loopback HTTP."""

import argparse
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager


def docker(*args, timeout=90, check=True):
    result = subprocess.run(
        ["docker", *args], text=True, capture_output=True, timeout=timeout, check=False
    )
    if check and result.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr or result.stdout}")
    return result.stdout.strip() if check else (result.stdout + result.stderr).strip()


@contextmanager
def container(image, volume, *command):
    name = f"rag-smoke-{uuid.uuid4().hex}"
    cid = docker(
        "create",
        "--pull=never",
        "--name",
        name,
        "--network",
        "none",
        "--mount",
        f"type=volume,source={volume},target=/app/data",
        "-e",
        "RAG_SMOKE_TEST_MODE=true",
        "-e",
        "OPENAI_API_KEY=",
        "-e",
        "ANTHROPIC_API_KEY=",
        image,
        *command,
    )
    try:
        docker("start", cid)
        yield cid
        docker("stop", "--time", "10", cid)
    except BaseException:
        print(
            docker("inspect", "--format", "{{json .State}}", cid, check=False),
            file=sys.stderr,
        )
        print(docker("logs", "--tail", "80", cid, check=False), file=sys.stderr)
        raise
    finally:
        docker("rm", "-f", cid)


HTTP = """
import json
import urllib.request
from urllib.error import HTTPError
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = 'http://127.0.0.1:8000'
def get(path):
    with opener.open(base + path, timeout=4) as response:
        return json.load(response)
"""


def wait_for_api(cid):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if docker("inspect", "--format", "{{.State.Running}}", cid) != "true":
            raise RuntimeError("API container exited during startup")
        try:
            docker("exec", cid, "python", "-c", HTTP + "get('/live')", timeout=10)
            return
        except (RuntimeError, subprocess.TimeoutExpired):
            time.sleep(1)
    raise RuntimeError("Loopback liveness probe timed out after 120 seconds")


CHECK_API = """
assert get('/live') == {'status': 'ok'}
health = get('/health')
assert health['status'] == 'ok' and health['timestamp'] > 0
assert '/query' in get('/openapi.json')['paths']
assert get('/ready') == dict(mode='smoke', database_available=True,
                            providers_configured=False, rag_ready=False)
try:
    opener.open(urllib.request.Request(base + '/query',
        data=b'{"query":"offline check"}', headers={'Content-Type':'application/json'}), timeout=4)
except HTTPError as exc:
    assert exc.code == 503
    assert json.load(exc) == {'detail':'AI operations unavailable in smoke-test mode'}
else:
    raise RuntimeError('Expected HTTP 503 from /query')
"""

CHECK_RUNTIME = """
import os
from pathlib import Path
from tempfile import TemporaryFile
import tiktoken
assert os.geteuid() == 10001
uid_line = next(line for line in Path('/proc/1/status').read_text().splitlines()
                if line.startswith('Uid:'))
assert list(map(int, uid_line.split()[1:])) == [10001] * 4, uid_line
assert not Path('/app/.env').exists()
assert not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY')
enc = tiktoken.get_encoding('cl100k_base')
assert enc.decode(enc.encode('offline check')) == 'offline check'
with TemporaryFile(dir='/app/data') as probe:
    probe.write(b'writable')
"""

SEED = """
from core.models import Chunk, EmbeddedChunk
from core.vector_store import VectorStore
store = VectorStore()
try:
    chunk = Chunk('smoke-id', 'synthetic', 'offline fixture', 0, 2, {'source':'smoke-fixture'})
    store.upsert([EmbeddedChunk(chunk, [1., 0.], 'synthetic')])
    assert store.query([1., 0.], top_k=1)[0].chunk.chunk_id == 'smoke-id'
finally:
    store.close()
"""


def verify(image):
    with ExitStack() as cleanup:
        volume = docker("volume", "create", "--label", "purpose=rag-smoke")
        cleanup.callback(docker, "volume", "rm", volume)
        with container(image, volume) as cid:
            wait_for_api(cid)
            docker("exec", cid, "python", "-c", HTTP + CHECK_API)
            docker("exec", cid, "python", "-c", CHECK_RUNTIME)
            docker("exec", cid, "python", "-m", "pip", "check")
        # Embedded Chroma gets exclusive access while the API is stopped.
        with container(image, volume, "python", "-c", SEED) as writer:
            if docker("wait", writer) != "0":
                raise RuntimeError("Synthetic Chroma write or vector lookup failed")
        with container(image, volume) as cid:
            wait_for_api(cid)
            docker(
                "exec",
                cid,
                "python",
                "-c",
                HTTP
                + CHECK_API
                + """
assert health['collection_size'] == 1
assert get('/sources') == {'sources':['smoke-fixture']}
""",
            )
    print("Docker smoke checks passed, including persistence after recreation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    try:
        verify(parser.parse_args().image)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
