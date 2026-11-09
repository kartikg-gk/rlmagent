import json
import os
import subprocess
import sys

from rlmagent.jobs import Jobs


async def test_ordinary_job_stops_when_its_owner_process_crashes(tmp_path):
    code = """
import asyncio, json, os, sys
from pathlib import Path
from rlmagent.jobs import Jobs
async def main():
    jobs = Jobs(sys.argv[1], Path(sys.argv[1]) / 'jobs')
    started = await jobs.job_start(argv=[sys.executable, '-c', 'import time; time.sleep(60)'])
    print(json.dumps(started), flush=True)
asyncio.run(main())
os._exit(0)
"""
    reply = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
        env=dict(os.environ), capture_output=True, text=True, timeout=30)
    assert reply.returncode == 0, reply.stderr
    key = json.loads(reply.stdout)["id"]
    reopened = Jobs(str(tmp_path), tmp_path / "jobs")
    try:
        done = await reopened.job_wait(key, 20)
        assert done["done"] and done["interrupted"] and done["stopped"]
    finally:
        await reopened.close()


async def test_bounded_log_unicode_cursors_and_reopened_metadata(tmp_path):
    jobs = Jobs(str(tmp_path), tmp_path / "jobs", max_log_bytes=200000)
    try:
        code = "import sys; sys.stdout.write('界'*100000)"
        started = await jobs.job_start(argv=[sys.executable, "-c", code])
        done = await jobs.job_wait(started["id"], 30)
        assert done["exit"] == 0 and done["truncated"]
        assert done["bytes"] == 199998 and done["total_bytes"] == 300000
        text, cursor = "", 0
        while True:
            part = await jobs.job_read(started["id"], cursor)
            assert part["cursor"] >= cursor
            text += part["text"]
            cursor = part["cursor"]
            if not part["more"]:
                break
        assert text.startswith("界" * 66000)
        assert "�" not in text
        reopened = Jobs(str(tmp_path), jobs.folder)
        assert (await reopened.jobs_list())[0]["total_bytes"] == 300000
    finally:
        await jobs.close()


async def test_explicit_persistent_service_survives_close_and_reconnect(tmp_path):
    jobs = Jobs(str(tmp_path), tmp_path / "jobs")
    command = f'"{sys.executable.replace(chr(92), "/")}" -u -c "import time; print(\'ready\'); time.sleep(60)"'
    svc = await jobs.service_start("server", command, ready="ready", wait=20, persist=True)
    assert svc["ready"] and svc["persistent"]
    await jobs.close()
    reopened = Jobs(str(tmp_path), jobs.folder)
    try:
        assert not (await reopened.jobs_list())[0]["done"]
        assert (await reopened.job_read("server"))["text"].strip() == "ready"
        stopped = await reopened.job_stop("server")
        assert stopped["done"] and stopped["stopped"]
    finally:
        for job in reopened.jobs.values():
            job.persistent = False
        await reopened.close()


async def test_job_environment_excludes_host_credentials_but_accepts_explicit_values(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-host-value")
    monkeypatch.setenv("RLM_AGENT_CONTROL_SOCKET", "private-control")
    jobs = Jobs(str(tmp_path), tmp_path / "jobs", env={"AGENT_COLOR": "blue"})
    try:
        started = await jobs.job_start(argv=[sys.executable, "-c",
            "import os,json; print(json.dumps({k:os.getenv(k) for k in "
            "['OPENAI_API_KEY','RLM_AGENT_CONTROL_SOCKET','AGENT_COLOR']}))"])
        await jobs.job_wait(started["id"], 30)
        values = json.loads((await jobs.job_read(started["id"]))["text"])
        assert values == {"OPENAI_API_KEY": None, "RLM_AGENT_CONTROL_SOCKET": None,
                          "AGENT_COLOR": "blue"}
    finally:
        await jobs.close()
