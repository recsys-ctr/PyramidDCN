"""Mandatory status-only eighteen-fit guard. This module opens no metric files."""
import json
from pathlib import Path

DATASETS = ("criteo", "avazu")
SEEDS = (2032, 2033, 2034)


def require_all_queues(original, native=None):
    original = Path(original).resolve()
    native = Path(native).resolve() if native is not None else original.parent / "native_input_training"
    if original.name != "robustness_training" or native.name != "native_input_training" or original.parent != native.parent:
        raise ValueError("Use sibling robustness_training and native_input_training queues")
    gate_file = original.parent / "robustness_inspection_gate.json"
    if not gate_file.is_file():
        raise ValueError("Mandatory eighteen-fit inspection gate is missing")
    gate = json.loads(gate_file.read_text())
    expected = [{"directory": "robustness_training", "planned_runs": 12},
                {"directory": "native_input_training", "planned_runs": 6}]
    if (gate.get("require_all_queues_before_metric_access") is not True
            or gate.get("total_planned_runs") != 18 or gate.get("queues") != expected):
        raise ValueError("Mandatory eighteen-fit inspection gate mismatch")
    rosters = [
        {f"robustness_r2_{d}_{a}_lr3e4_ep2_s{s}" for d in DATASETS for s in SEEDS
         for a in ("selection", "learned_budget")},
        {f"native_field_{d}_fibinetpp_lr3e4_ep2_s{s}" for d in DATASETS for s in SEEDS},
    ]
    for directory, item, names in zip((original, native), expected, rosters):
        path = directory / "status.json"
        if not path.is_file():
            raise ValueError("All eighteen fits must finish before metric access")
        state = json.loads(path.read_text())
        complete = state.get("completed_runs", [])
        if (state.get("state") != "completed" or state.get("planned_runs") != item["planned_runs"]
                or len(complete) != item["planned_runs"] or state.get("failed_runs")
                or {x.get("run") for x in complete} != names
                or any(x.get("state") != "completed" for x in complete)):
            raise ValueError("Metric access refused: all original twelve plus native six must complete")
    return True
