import copy
import pytest

from rlmagent.kernel.wasm import WasmKernel, available
from rlmagent.contract import parse_contract, ContractError
from test_contract import VALID

pytestmark = pytest.mark.asyncio

@pytest.mark.skipif(not available(), reason='sandbox unavailable')
async def test_restore_plain_data_definitions_deletions_and_helpers(tmp_path):
    kernel = WasmKernel(str(tmp_path), {}, startup_code='def helper(): return 9', restore=True)
    try:
        out = await kernel.run("x = {'items': [1, 2]}\nold = 3\ndef twice(n): return n * 2")
        assert out.error is None, out.error
        assert 'helper' not in kernel._checkpoint['dropped']
        out = await kernel.run("del old\nx['items'].append(3)\nbad = (i for i in range(3))")
        assert out.error is None
        await kernel.interrupt()
        out = await kernel.run("print(x, twice(4), helper(), 'old' in globals(), 'bad' in globals())")
        assert out.error is None, out.error
        assert "{'items': [1, 2, 3]} 8 9 False False" in out.output
        assert 'unavailable: bad' in out.output
    finally:
        await kernel.shutdown()

@pytest.mark.skipif(not available(), reason='sandbox unavailable')
async def test_timeout_recovers_only_last_successful_checkpoint(tmp_path):
    kernel = WasmKernel(str(tmp_path), {}, timeout=10, restore=True)
    try:
        assert (await kernel.run('x = 42')).error is None
        kernel._runtime.timeout = 0.5
        out = await kernel.run('x = 99\nwhile True: pass')
        assert out.timed_out and out.restarted
        out = await kernel.run('print(x)')
        assert out.error is None, out.error
        assert out.output.endswith('42\n')
    finally:
        await kernel.shutdown()

@pytest.mark.skipif(not available(), reason='sandbox unavailable')
async def test_disabled_restore_keeps_no_checkpoint(tmp_path):
    kernel = WasmKernel(str(tmp_path), {})
    try:
        await kernel.run('x = 42')
        await kernel.interrupt()
        out = await kernel.run("print('x' in globals())")
        assert out.output == 'False\n'
        assert kernel._checkpoint == {}
    finally:
        await kernel.shutdown()

async def test_contract_restore_is_optional_and_typed():
    doc = copy.deepcopy(VALID)
    doc['tools'] = ['python', 'Read']
    doc['kernel'] = {'kind': 'wasm', 'approve': 'deny', 'network': False}
    assert not parse_contract(doc).kernel.restore
    doc['kernel']['restore'] = True
    assert parse_contract(doc).kernel.restore
    doc['kernel']['restore'] = 'yes'
    with pytest.raises(ContractError, match='kernel.restore'):
        parse_contract(doc)
    doc['kernel'].update(restore=True, kind='local')
    with pytest.raises(ContractError, match='kernel.restore'):
        parse_contract(doc)

async def test_cli_accepts_restore():
    from rlmagent.cli.main import _build_run_parser
    assert _build_run_parser().parse_args(['--restore-kernel']).restore_kernel
    assert not _build_run_parser().parse_args([]).restore_kernel
