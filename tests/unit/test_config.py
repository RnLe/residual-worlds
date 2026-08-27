"""Strict contract loading: acceptance, rejection, and cross-checks."""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from residual_worlds.config import ContractError, load_contract, load_strict_yaml
from residual_worlds.paths import repository_root

CONTRACT_PATH = repository_root() / "configs" / "experiment_contract.yaml"
SMOKE_PATH = repository_root() / "configs" / "smoke.yaml"


def _load_raw() -> dict[str, Any]:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_shipped_contract_loads() -> None:
    contract = load_contract(CONTRACT_PATH)
    assert contract.evaluation.primary_world == "composite_standard"
    assert contract.data.primary_budget_total == 2048
    assert contract.numerics.root_seed == 730241
    assert contract.planning.candidates == 256
    assert len(contract.unresolved_decisions) == 11


def test_shipped_smoke_config_loads() -> None:
    contract = load_contract(SMOKE_PATH)
    assert contract.data.primary_budget_total == 128
    assert contract.unresolved_decisions == ()


def test_duplicate_yaml_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text("a: 1\na: 2\n", encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate YAML key"):
        load_strict_yaml(path)


def test_nested_duplicate_yaml_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text("outer:\n  a: 1\n  a: 2\n", encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate YAML key"):
        load_strict_yaml(path)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    document = _load_raw()
    document["numerics"]["not_a_real_setting"] = 3
    with pytest.raises(ContractError, match="unknown keys"):
        load_contract(_write(tmp_path, document))


def test_missing_key_rejected(tmp_path: Path) -> None:
    document = _load_raw()
    del document["numerics"]["control_dt_s"]
    with pytest.raises(ContractError, match="missing key"):
        load_contract(_write(tmp_path, document))


def test_budget_not_multiple_of_units_rejected(tmp_path: Path) -> None:
    document = _load_raw()
    document["data"]["adaptation_budgets_total"] = [256, 1000, 2048, 8192, 16384]
    with pytest.raises(ContractError, match="four collection units"):
        load_contract(_write(tmp_path, document))


def test_primary_budget_must_be_listed(tmp_path: Path) -> None:
    document = _load_raw()
    document["data"]["primary_budget_total"] = 512
    with pytest.raises(ContractError, match="primary budget"):
        load_contract(_write(tmp_path, document))


def test_exploration_fractions_must_sum_to_one(tmp_path: Path) -> None:
    document = _load_raw()
    document["data"]["exploration"]["multisine_fraction"] = 0.5
    with pytest.raises(ContractError, match="sum to one"):
        load_contract(_write(tmp_path, document))


def test_component_equality_enforced(tmp_path: Path) -> None:
    document = _load_raw()
    document["worlds"]["payload_standard"]["payload_kg"] = 0.30
    with pytest.raises(ContractError, match="must equal its copy"):
        load_contract(_write(tmp_path, document))


def test_exactly_one_primary_world(tmp_path: Path) -> None:
    document = _load_raw()
    document["worlds"]["payload_standard"]["role"] = "primary"
    with pytest.raises(ContractError, match="exactly one world"):
        load_contract(_write(tmp_path, document))


def test_profile_values_must_match_selected_profile(tmp_path: Path) -> None:
    document = _load_raw()
    document["planning"]["candidates"] = 128
    with pytest.raises(ContractError, match="selected profile"):
        load_contract(_write(tmp_path, document))


def test_contrast_weights_must_sum_to_zero(tmp_path: Path) -> None:
    document = _load_raw()
    document["analysis"]["primary_contrast"]["weights"] = {"residual": 1.0, "blackbox": -0.5}
    with pytest.raises(ContractError, match="sum to zero"):
        load_contract(_write(tmp_path, document))


def test_unknown_contrast_method_rejected(tmp_path: Path) -> None:
    document = _load_raw()
    document["analysis"]["primary_contrast"]["weights"] = {"residual": 1.0, "gpt": -1.0}
    with pytest.raises(ContractError, match="unknown methods"):
        load_contract(_write(tmp_path, document))


def test_residual_zero_init_is_mandatory(tmp_path: Path) -> None:
    document = _load_raw()
    document["models"]["residual"]["final_layer_zero_init"] = False
    with pytest.raises(ContractError, match="zero-initialized"):
        load_contract(_write(tmp_path, document))


def test_horizon_knot_divisibility(tmp_path: Path) -> None:
    document = _load_raw()
    document["planning"]["action_knots"] = 7
    with pytest.raises(ContractError, match="whole multiple"):
        load_contract(_write(tmp_path, document))


def test_contract_is_deeply_frozen() -> None:
    contract = load_contract(CONTRACT_PATH)
    with pytest.raises(AttributeError):
        contract.numerics.control_dt_s = 0.1  # type: ignore[misc]


def test_copy_of_contract_equal(tmp_path: Path) -> None:
    # Serialization stability: reloading the same file gives equal values.
    a = load_contract(CONTRACT_PATH)
    b = load_contract(CONTRACT_PATH)
    assert copy.deepcopy(a.arm) == b.arm
    assert a.task.cost == b.task.cost
