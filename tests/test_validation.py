from predictive_cyberattack.config import load_config
from predictive_cyberattack.spark.validation import required_input_columns


def test_required_input_columns_are_derived_from_config():
    config = load_config("configs/demo.yaml")
    columns = required_input_columns(config)

    assert "state" in columns
    assert "dtrate" in columns
    assert "label" in columns
    assert "attack_cat_index" not in columns
