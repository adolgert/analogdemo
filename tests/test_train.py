import torch

from analogdemo.model import build_model
from analogdemo.train import (
    _reconcile_metrics,
    build_optimizer,
    build_scheduler,
    parameter_groups,
    train_batch,
)


def config(**updates):
    result = {"learning_rate": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
              "warmup_epochs": 5, "epochs": 150}
    result.update(updates)
    return result


def test_parameter_groups_exclude_bias_and_bn_from_decay():
    model = build_model()
    groups = parameter_groups(model, 5e-4)
    decayed = {id(p) for p in groups[0]["params"]}
    for name, parameter in model.named_parameters():
        assert (id(parameter) in decayed) == (parameter.ndim != 1 and not name.endswith(".bias"))


def test_sam_updates_batch_norm_statistics_once():
    torch.manual_seed(3)
    model = build_model()
    optimizer = build_optimizer(model, config())
    before = [int(bn.num_batches_tracked) for bn in model.bns]
    train_batch(model, torch.randn(2, 3, 32, 32), torch.tensor([1, 2]), optimizer, "sam", 0.05)
    after = [int(bn.num_batches_tracked) for bn in model.bns]
    assert after == [value + 1 for value in before]


def test_full_schedule_is_preserved_for_pilot():
    model = build_model()
    optimizer = build_optimizer(model, config())
    scheduler = build_scheduler(optimizer, config())
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    assert scheduler.state_dict()["last_epoch"] == 3
    assert optimizer.param_groups[0]["lr"] < 0.1


def test_resume_reconciles_metrics_with_durable_checkpoint(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"epoch": 0, "value": "old"}\n'
        '{"epoch": 1, "value": "first"}\n'
        '{"epoch": 1, "value": "duplicate"}\n'
        '{"epoch": 2, "value": "not checkpointed"}\n'
    )
    _reconcile_metrics(path, 1)
    assert path.read_text().splitlines() == [
        '{"epoch": 0, "value": "old"}',
        '{"epoch": 1, "value": "duplicate"}',
    ]


def test_resume_discards_crash_truncated_final_metric(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 0, "value": "durable"}\n{"epoch": 1, "value":')
    _reconcile_metrics(path, 0)
    assert path.read_text() == '{"epoch": 0, "value": "durable"}\n'
