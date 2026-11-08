import datetime
import ipaddress
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest


def test_extra_ca_accepts_local_tls_without_disabling_verification(tmp_path, monkeypatch):
    cryptography = pytest.importorskip("cryptography.x509")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    from rlmagent.model.transport.client import certificate_context

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = cryptography.Name([cryptography.NameAttribute(NameOID.COMMON_NAME, "test-local")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        cryptography.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(cryptography.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(cryptography.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            cryptography.SubjectAlternativeName(
                [cryptography.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    ca = tmp_path / "fixture.pem"
    private = tmp_path / "fixture.key"
    ca.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"trusted")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(ca, private)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.delenv("NODE_EXTRA_CA_CERTS", raising=False)
        url = f"https://127.0.0.1:{server.server_port}"
        with httpx.Client(verify=certificate_context(), trust_env=False) as client:
            with pytest.raises(httpx.ConnectError):
                client.get(url)
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(ca))
        verification = certificate_context()
        assert verification.check_hostname and verification.verify_mode == ssl.CERT_REQUIRED
        with httpx.Client(verify=verification, trust_env=False) as client:
            assert client.get(url).text == "trusted"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
