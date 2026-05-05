from __future__ import annotations

import atexit
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest


def _ensure_hotvect_jars_present() -> None:
    jar_dir = Path(__file__).resolve().parents[2] / "hotvectjar"
    jar_dir.mkdir(parents=True, exist_ok=True)
    required_patterns = {
        "hotvect-offline-util-*-jar-with-dependencies.jar": "hotvect-offline-util-test-jar-with-dependencies.jar",
        "hotvect-algorithm-demo-*-jar-with-dependencies.jar": "hotvect-algorithm-demo-test-jar-with-dependencies.jar",
    }
    for pattern, filename in required_patterns.items():
        existing = list(jar_dir.glob(pattern))
        if len(existing) == 0:
            dummy = jar_dir / filename
            dummy.touch(exist_ok=True)
            atexit.register(lambda path=dummy: path.unlink(missing_ok=True))


def test_sagemaker_pipeline_params_include_encode_and_audit_flags() -> None:
    _ensure_hotvect_jars_present()
    from hotvect.sagemaker import ALGO_PIPELINE_HYPERPARAMETER_PREFIX, SagemakerTrainingExecutor

    def _eval(_: str) -> dict:
        return {"ok": True}

    executor = SagemakerTrainingExecutor.__new__(SagemakerTrainingExecutor)
    executor.training_job_definition = {"HyperParameters": {}}
    executor.algorithm_pipeline = SimpleNamespace(
        last_test_time=date(2026, 1, 1),
        evaluation_function=_eval,
        parameter_version="pv",
        execute_performance_test=False,
        encode_test_data=True,
        execute_audit=True,
    )

    executor._add_algorithm_pipeline_params_as_hyperparameters()

    payload = json.loads(executor.hyperparameters[ALGO_PIPELINE_HYPERPARAMETER_PREFIX])
    assert "evaluation_func" not in payload
    assert payload["execute_performance_test"] is False
    assert payload["encode_test_data"] is True
    assert payload["execute_audit"] is True


@pytest.mark.parametrize(
    ("jvm_options", "expected"),
    [
        ([], ["-XX:MaxRAMPercentage=80"]),
        (["-Dfoo=bar"], ["-Dfoo=bar", "-XX:MaxRAMPercentage=80"]),
    ],
)
def test_sagemaker_rebuilder_normalizes_jvm_options_to_include_default_heap_cap(
    monkeypatch, tmp_path, jvm_options, expected
) -> None:
    _ensure_hotvect_jars_present()
    from hotvect.sagemaker import ALGO_PIPELINE_CONTEXT_PREFIX, SagemakerAlgorithmPipelineRebuilder

    rebuilder = SagemakerAlgorithmPipelineRebuilder.__new__(SagemakerAlgorithmPipelineRebuilder)
    rebuilder.sagemaker_env = SimpleNamespace(
        input_dir=str(tmp_path / "input"),
        hyperparameters={ALGO_PIPELINE_CONTEXT_PREFIX: {"jvm_options": jvm_options}},
    )
    monkeypatch.setattr(rebuilder, "_get_metadata_dir", lambda: str(tmp_path / "meta-root"))
    monkeypatch.setattr(rebuilder, "_get_output_dir", lambda: str(tmp_path / "out-root"))

    context = rebuilder._rebuild_algorithm_pipeline_context(tmp_path / "algo.jar")

    assert context.jvm_options == expected
