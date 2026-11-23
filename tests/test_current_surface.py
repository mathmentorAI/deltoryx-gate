"""Engineering checks for the sole supported package and CLI surface.

No model is executed or scored here. Product evaluations use live NVIDIA calls.
"""
import importlib.util
import json

import pytest
import claimlayer
import deltoryx_gate
from claimlayer.v4 import planner


RETIRED_MODULES = (
    'claimlayer.ledger', 'claimlayer.verify', 'claimlayer.receipt', 'claimlayer.mcp_server',
    'claimlayer.v4.legacy_derivation', 'claimlayer.v4.authorization', 'claimlayer.v4.reader',
    'claimlayer.v4.repair', 'claimlayer.v4.triage', 'claimlayer.semantic.engine',
)


def test_public_imports_point_to_one_implementation():
    assert claimlayer.__version__ == deltoryx_gate.__version__ == '4.0.0'
    assert claimlayer.CONTRACT_VERSION == 'gate-layers-1'
    assert claimlayer.verify_claim is deltoryx_gate.verify_claim is planner.verify_claim
    for name in ('Ledger', 'verify', 'verify_plan', 'review_response', 'EvidenceLayer', 'MCPVerificationServer'):
        assert not hasattr(claimlayer, name)
        assert not hasattr(claimlayer.v4, name)


@pytest.mark.parametrize('name', RETIRED_MODULES)
def test_retired_modules_are_not_distributed(name):
    assert importlib.util.find_spec(name) is None


def test_cli_has_only_live_verify_claim_command():
    from claimlayer.cli import main
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    for name in ('verify', 'demo', 'mcp'):
        with pytest.raises(SystemExit) as exc:
            main([name])
        assert exc.value.code == 2


def test_cli_missing_key_cannot_run_an_alternate_engine(monkeypatch, capsys):
    from claimlayer.cli import main
    monkeypatch.delenv('GATE_TEST_UNCONFIGURED_KEY', raising=False)
    code = main(['verify-claim', '--claim', 'a claim', '--source', 'doc=source',
                 '--api-key-env', 'GATE_TEST_UNCONFIGURED_KEY'])
    assert code == 3
    result = json.loads(capsys.readouterr().err)
    assert result['stage'] == 'infrastructure'
    assert result['code'] == 'MISSING_PROVIDER_API_KEY'
