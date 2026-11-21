"""Server web Memori tidak boleh berbagi port dengan proses lain (di Windows SO_REUSEADDR mengizinkannya tanpa galat)."""
import os
from http.server import BaseHTTPRequestHandler

import pytest

from conftest import load_module


@pytest.fixture(scope="module")
def web():
    return load_module("memory_web_under_test", "tools/memory_web.py")


def test_reuse_address_disabled_on_windows(web):
    if os.name == "nt":
        assert web.ExclusiveServer.allow_reuse_address is False
    assert web.ExclusiveServer.daemon_threads is True


def test_second_server_on_same_port_fails_instead_of_sharing_it(web):
    first = web.ExclusiveServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        port = first.server_address[1]
        with pytest.raises(OSError):
            web.ExclusiveServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    finally:
        first.server_close()


def test_note_hints_unchanged_by_the_server_refactor(web):
    assert web.note_hints("Dia pindah kemarin.")  # kata ganti di awal + waktu relatif
    assert web.note_hints("Bu Rina tinggal di Jalan Mawar nomor 5.") == []
