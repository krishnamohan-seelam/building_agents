"""Validates recorder/samples/*.json against recorder/trace.schema.json (M1)."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

RECORDER_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = RECORDER_DIR / "trace.schema.json"
SAMPLES_DIR = RECORDER_DIR / "samples"


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_schema_itself_is_valid(schema):
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    "sample_path", sorted(SAMPLES_DIR.glob("*.json")), ids=lambda p: p.name
)
def test_sample_trace_validates(schema, sample_path):
    with open(sample_path, encoding="utf-8") as f:
        trace = json.load(f)

    Draft202012Validator(schema).validate(trace)
