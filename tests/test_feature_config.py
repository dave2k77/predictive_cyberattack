from predictive_cyberattack.config import load_config


def test_binary_label_is_not_configured_as_input_feature():
    config = load_config("configs/demo.yaml")
    feature_columns = set(config.features["categorical"]) | set(config.features["numeric"])

    assert config.features["label"] not in feature_columns
