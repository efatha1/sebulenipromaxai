"""Unit tests for U11 API validation and structured errors."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
instrument:
  instrument_id: TEST_INSTRUMENT
data_source:
  ohlc_path: data/test.csv
time:
  source_timezone: UTC
  runtime_timezone: UTC
  daily_close:
    time: "17:00"
    timezone: America/New_York
sessions:
  calendar_name: default
  weekend_policy: include
  holiday_policy: include
  definitions:
    - name: primary
      start: "00:00"
      end: "23:59"
resampling:
  base_timeframe: 1m
  target_timeframes: ["5m", "15m", "1h", "4h", "1d"]
features:
  enabled_features: ["returns", "calendar_time"]
  deterministic_derived_features: ["returns", "calendar_time"]
labeling:
  thresholds: [10.0]
  horizon_bars: [3, 5]
walk_forward:
  train_bars: 4
  validation_bars: 2
  test_bars: 2
  step_bars: 2
training:
  random_seed: 7
  batch_size: 2
  learning_rate: 0.001
  max_epochs: 1
  device_preference: cpu
  allow_nondeterministic: false
retrieval:
  top_k_analogs: 2
api:
  host: 127.0.0.1
  port: 8000
reporting:
  output_dir: "%s"
schedules:
  reporting_cron: "0 9 * * *"
  retraining_review_cron: "0 0 * * 1"
"""
        % str(tmp_path).replace("\\", "/"),
        encoding="utf-8",
    )
    return path


def test_u11_predict_validation_returns_structured_errors(tmp_path: Path) -> None:
    app = create_app(config_path=_write_config(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "instrument_id": "TEST_INSTRUMENT",
            "bars_1m": [],
            "horizon_mode": "single",
            "horizon_bars": 3,
            "threshold": 10.0,
            "top_k_analogs": 1,
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert "errors" in payload
    assert payload["errors"][0]["location"]
    assert payload["errors"][0]["message"]
