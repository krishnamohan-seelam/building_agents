"""Checks query_decomposer/example.yaml has the shape recorder/record.py (M2) expects."""

from pathlib import Path

import yaml

EXAMPLE_YAML_PATH = Path(__file__).resolve().parent.parent / "example.yaml"


def test_example_yaml_has_required_keys():
    with open(EXAMPLE_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["title"]
    assert config["slug"]
    assert config["post_url"]


def test_example_yaml_has_one_to_three_sample_inputs():
    with open(EXAMPLE_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sample_inputs = config["sample_inputs"]
    assert 1 <= len(sample_inputs) <= 3
    assert all(isinstance(q, str) and q.strip() for q in sample_inputs)
