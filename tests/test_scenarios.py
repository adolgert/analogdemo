import pytest

from analogdemo.scenarios import AFFINE_LAYERS, demonstration_scenarios, noise_config


def test_scenario_grid_is_deterministic_and_complete():
    first = demonstration_scenarios()
    second = demonstration_scenarios()
    assert first == second
    assert len(first) == 3 * 5 * 5
    assert len({scenario["id"] for scenario in first}) == len(first)


def test_independent_and_shared_keep_same_transient_amplitude():
    independent = noise_config(mechanism="independent_transient", location="early", amplitude=0.1)
    shared = noise_config(
        mechanism="spatially_shared_transient", location="early", amplitude=0.1, spatial_rho=0.75
    )
    assert independent["transient"] == shared["transient"]
    assert independent["spatial_rho"] == 0.0
    assert shared["spatial_rho"] == {"conv1": 0.75, "conv2": 0.75}


def test_locations_and_persistent_lifetime_budget():
    scenarios = demonstration_scenarios(amplitudes=(0.03,))
    all_weight = next(
        item for item in scenarios if item["mechanism"] == "persistent_weight" and item["location"] == "all"
    )
    assert tuple(all_weight["noise_config"]["weight"]) == AFFINE_LAYERS
    assert set(all_weight["noise_config"]["weight"].values()) == {0.03}
    assert all_weight["sampling"] == {"chips": 32, "reads_per_chip": 1}


def test_location_comparison_does_not_confound_layer_count():
    early = noise_config(mechanism="independent_transient", location="early", amplitude=0.1)
    middle = noise_config(mechanism="independent_transient", location="middle", amplitude=0.1)
    late = noise_config(mechanism="independent_transient", location="late", amplitude=0.1)
    for config in (early, middle, late):
        assert sum(value > 0 for value in config["transient"].values()) == 2
    head = noise_config(mechanism="independent_transient", location="head", amplitude=0.1)
    assert {name for name, value in head["transient"].items() if value > 0} == {"classifier"}


def test_invalid_scenario_inputs_fail():
    with pytest.raises(ValueError):
        noise_config(mechanism="bogus", location="all", amplitude=0.1)
    with pytest.raises(ValueError):
        noise_config(mechanism="independent_transient", location="nowhere", amplitude=0.1)
    with pytest.raises(ValueError):
        noise_config(mechanism="independent_transient", location="all", amplitude=-1)
