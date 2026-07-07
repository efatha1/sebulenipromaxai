"""Unit tests for U11 CLI parsing surfaces."""

from __future__ import annotations

from typer.testing import CliRunner

from training.cli import app


def test_u11_cli_help_exposes_required_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout
    assert "generate-report" in result.stdout
    assert "request-retraining" in result.stdout
