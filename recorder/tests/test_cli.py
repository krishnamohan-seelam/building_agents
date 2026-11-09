"""Smoke test for recorder/cli.py (M2-T9): argparse wiring only, record_run is mocked."""

from unittest.mock import patch

import pytest

from recorder.cli import main


@patch("recorder.cli.record_run")
def test_cli_wires_input_and_cached_through_to_record_run(mock_record_run):
    mock_record_run.return_value = {"example": "query_decomposer", "steps": []}

    main(["query_decomposer", "--input", "2", "--cached"])

    mock_record_run.assert_called_once_with(
        "query_decomposer", input_index=2, cached=True, preview_path=None
    )


@patch("recorder.cli.record_run")
def test_cli_defaults_cached_to_false(mock_record_run):
    mock_record_run.return_value = {"example": "query_decomposer", "steps": []}

    main(["query_decomposer", "--input", "1"])

    mock_record_run.assert_called_once_with(
        "query_decomposer", input_index=1, cached=False, preview_path=None
    )


@patch("recorder.cli.record_run")
def test_cli_wires_preview_path_through_to_record_run(mock_record_run):
    mock_record_run.return_value = {"example": "query_decomposer", "steps": []}

    main(["query_decomposer", "--input", "1", "--preview", "../my-blog"])

    mock_record_run.assert_called_once_with(
        "query_decomposer", input_index=1, cached=False, preview_path="../my-blog"
    )


def test_cli_publish_flag_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        main(["query_decomposer", "--input", "1", "--publish", "../my-blog"])
