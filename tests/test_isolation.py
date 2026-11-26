import socket

import httpx
import pytest


def test_external_http_and_udp_are_blocked_before_io():
    with (
        httpx.Client(trust_env=False) as client,
        pytest.raises(pytest.fail.Exception, match="Network access blocked"),
    ):
        client.get("https://example.invalid")
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(pytest.fail.Exception, match="blocked"),
    ):
        sock.sendto(b"test", ("192.0.2.1", 9))


def test_local_event_loop_socket_pair():
    reader, writer = socket.socketpair()
    with reader, writer:
        writer.sendall(b"test")
        assert reader.recv(4) == b"test"
