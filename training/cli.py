"""U11 CLI workflows for training, prediction, and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from api.dependencies import AppServices, serialize_cli_error
from api.schemas import PredictRequest
from inference.reporting import serialize_prediction_response
from training.contracts import PredictionResponseContract

app = typer.Typer(help="Sebuleni Pro Max AI CLI")


@app.command("train")
def train_command(
    config_path: Path = typer.Option(..., exists=True),
    bundle_path: Path = typer.Option(..., exists=True),
    evaluation_output_path: Path = typer.Option(None),
    training_input_root: Path = typer.Option(None),
    training_output_root: Path = typer.Option(None),
) -> None:
    """Run training/evaluation from a serialized U9 training bundle."""
    services = AppServices(config_path=config_path)
    try:
        result = services.run_training_bundle(
            bundle_path=bundle_path,
            evaluation_output_path=evaluation_output_path,
            input_root=training_input_root,
            output_root=training_output_root,
        )
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)


@app.command("train-sharded")
def train_sharded_command(
    config_path: Path = typer.Option(..., exists=True),
    manifest_path: Path = typer.Option(..., exists=True),
    evaluation_output_path: Path = typer.Option(None),
    training_input_root: Path = typer.Option(None),
    training_output_root: Path = typer.Option(None),
) -> None:
    """Run training/evaluation from sharded preprocessing outputs."""
    services = AppServices(config_path=config_path)
    try:
        result = services.run_training_sharded_manifest(
            manifest_path=manifest_path,
            evaluation_output_path=evaluation_output_path,
            input_root=training_input_root,
            output_root=training_output_root,
        )
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)



@app.command("evaluate")
def evaluate_command(
    config_path: Path = typer.Option(..., exists=True),
    bundle_path: Path = typer.Option(..., exists=True),
    evaluation_output_path: Path = typer.Option(None),
    training_input_root: Path = typer.Option(None),
    training_output_root: Path = typer.Option(None),
) -> None:
    """Run walk-forward evaluation from the same serialized training bundle."""
    services = AppServices(config_path=config_path)
    try:
        result = services.run_training_bundle(
            bundle_path=bundle_path,
            evaluation_output_path=evaluation_output_path,
            input_root=training_input_root,
            output_root=training_output_root,
        )
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)


@app.command("evaluate-sharded")
def evaluate_sharded_command(
    config_path: Path = typer.Option(..., exists=True),
    manifest_path: Path = typer.Option(..., exists=True),
    evaluation_output_path: Path = typer.Option(None),
    training_input_root: Path = typer.Option(None),
    training_output_root: Path = typer.Option(None),
) -> None:
    """Run walk-forward evaluation from sharded preprocessing outputs."""
    services = AppServices(config_path=config_path)
    try:
        result = services.run_training_sharded_manifest(
            manifest_path=manifest_path,
            evaluation_output_path=evaluation_output_path,
            input_root=training_input_root,
            output_root=training_output_root,
        )
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)


@app.command("predict")
def predict_command(
    config_path: Path = typer.Option(..., exists=True),
    active_model_manifest_path: Path = typer.Option(..., exists=True),
    request_path: Path = typer.Option(..., exists=True),
) -> None:
    """Run prediction using the shared U10 inference contract."""
    services = AppServices(
        config_path=config_path,
        active_model_manifest_path=active_model_manifest_path,
    )
    try:
        request = PredictRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
        responses = services.predict_request(request)
        payload = [serialize_prediction_response(item) for item in responses]
        typer.echo(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)


@app.command("generate-report")
def generate_report_command(
    config_path: Path = typer.Option(..., exists=True),
    responses_path: Path = typer.Option(..., exists=True),
    report_name: str | None = typer.Option(None),
) -> None:
    """Generate a batch report from approved prediction outputs."""
    services = AppServices(config_path=config_path)
    try:
        payload = json.loads(responses_path.read_text(encoding="utf-8"))
        responses = payload if isinstance(payload, list) else [payload]
        parsed_responses = tuple(
            PredictionResponseContract.model_validate(item)
            for item in responses
        )
        artifact = services.generate_prediction_report(parsed_responses, report_name=report_name)
        typer.echo(
            json.dumps(
                {
                    "report_id": artifact.report_id,
                    "report_type": artifact.report_type,
                    "output_path": str(artifact.output_path),
                    "generated_at": artifact.generated_at.isoformat(),
                    "summary": artifact.summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)


@app.command("request-retraining")
def request_retraining_command(
    config_path: Path = typer.Option(..., exists=True),
    reason: str = typer.Option(...),
    candidate_model_id: str | None = typer.Option(None),
) -> None:
    """Create a retraining request artifact."""
    services = AppServices(config_path=config_path)
    try:
        result = services.create_retraining_request(
            reason=reason,
            candidate_model_id=candidate_model_id,
        )
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)


@app.command("inspect-model")
def inspect_model_command(
    config_path: Path = typer.Option(..., exists=True),
    active_model_manifest_path: Path = typer.Option(..., exists=True),
) -> None:
    """Inspect the current active model through the shared adapter."""
    services = AppServices(
        config_path=config_path,
        active_model_manifest_path=active_model_manifest_path,
    )
    try:
        typer.echo(json.dumps(services.inspect_current_model(), indent=2, sort_keys=True))
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        typer.echo(json.dumps(serialize_cli_error(exc), indent=2, sort_keys=True))
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
