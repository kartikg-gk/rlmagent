"""Google cloud authentication for the compatible chat endpoint."""

from __future__ import annotations

import asyncio
import os
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

    def _load(self):
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError:
            raise ConfigError(
                "Install rlmagent[vertex] to use Google application credentials"
            ) from None
        if self.credentials is None:
            self.credentials, detected = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.project = self.project or detected
        if not self.project:
            raise ConfigError("Set GOOGLE_CLOUD_PROJECT for the cloud endpoint")
        if not self.credentials.valid:
            self.credentials.refresh(Request())
        project, location = quote(self.project, safe=""), quote(self.location, safe="")
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        return Credential(
            self.credentials.token,
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
