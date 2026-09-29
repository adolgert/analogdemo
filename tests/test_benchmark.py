import torch

import analogdemo.benchmark as benchmark_module
from analogdemo.calibration import Calibration


LAYERS = tuple([f"conv{i}" for i in range(1, 7)] + ["classifier"])


def test_sampling_benchmark_uses_equal_main_budgets(monkeypatch):
    calls = []
    calibration = Calibration(
        1, {name: 1.0 for name in LAYERS}, {name: 1.0 for name in LAYERS}, 16
    )

    monkeypatch.setattr(benchmark_module, "select_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(benchmark_module, "environment_manifest", lambda: {"test": True})
    monkeypatch.setattr(benchmark_module, "synchronize", lambda _device: None)
    monkeypatch.setattr(benchmark_module, "calibrate_model", lambda *_args, **_kwargs: calibration)

    def sample(_model, images, _ids, _config, _calibration, *, num_chips, num_reads, **_kwargs):
        calls.append((num_chips, num_reads, len(images)))
        return torch.zeros(num_chips, num_reads, len(images), 10)

    monkeypatch.setattr(benchmark_module, "sample_logits", sample)
    result = benchmark_module.benchmark_sampling("cpu")

    assert calls == [(1, 1, 1), (1, 4, 16), (1, 4, 16), (4, 1, 16), (1, 64, 1)]
    assert {row["realizations"] for row in result["main_workloads"]} == {4}
    assert {row["network_executions"] for row in result["main_workloads"]} == {64}
    assert result["selected_image_workload"]["network_executions"] == 64
