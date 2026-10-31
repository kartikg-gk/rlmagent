"""Google cloud authentication for the compatible chat endpoint."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from urllib.parse import quote

from rlmagent.model.settings import ConfigError, Credential, OpenAIProfile


class ApplicationCredentials:
    def __init__(self):
        self.credentials = None
        self.project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("CLOUDSDK_CORE_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
        )
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        self._lock = asyncio.Lock()
        self._token = None
        self._expires = 0

    def _gcloud(self):
        if not self.project:
            raise ConfigError("Set GOOGLE_CLOUD_PROJECT for the cloud endpoint")
        if self._token is None or time.monotonic() >= self._expires:
            executable = shutil.which("gcloud")
            if executable is None:
                raise ConfigError("Configure application credentials or install gcloud")
            result = subprocess.run(
                [executable, "auth", "application-default", "print-access-token"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise ConfigError("Cloud login failed; run gcloud auth application-default login")
            self._token = result.stdout.strip()
            self._expires = time.monotonic() + 3500
        return self._endpoint(self._token)

    def _load(self):
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError:
            return self._gcloud()
        if self.credentials is None:
            try:
                self.credentials, detected = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except google.auth.exceptions.DefaultCredentialsError:
                return self._gcloud()
            self.project = self.project or detected
        if not self.project:
            raise ConfigError("Set GOOGLE_CLOUD_PROJECT for the cloud endpoint")
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        return self._endpoint(self.credentials.token)

    def _endpoint(self, token):
        project, location = quote(self.project, safe=""), quote(self.location, safe="")
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        return Credential(
            token,
            f"https://{host}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi",
        )

    async def __call__(self):
        async with self._lock:
            return await asyncio.to_thread(self._load)


def build_vertex():
    from rlmagent.model.oai_compatible import OpenAIProvider

    resolver = ApplicationCredentials()
    return OpenAIProvider(
        OpenAIProfile(name="vertex", credential=Credential(None, ""), resolver=resolver)
    )
