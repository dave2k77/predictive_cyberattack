from predictive_cyberattack.config import load_config


def test_demo_config_loads_expected_paths():
    config = load_config("configs/demo.yaml")

    assert config.mode == "demo"
    assert config.paths["raw_csv"] == "data/sample/unsw_sample.csv"
    assert config.training["pca_components"] == 5
