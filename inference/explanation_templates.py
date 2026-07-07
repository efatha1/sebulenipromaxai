"""Deterministic grounded explanation templates for U8."""

from __future__ import annotations

from training.contracts import AnalogRecordContract, PredictionRecordContract


def build_grounded_explanation_text(
    *,
    prediction: PredictionRecordContract,
    analogs: tuple[AnalogRecordContract, ...],
    summary_statistics: dict[str, float],
    requested_top_k: int,
) -> str:
    """Build a deterministic short grounded explanation string.

    Args:
        prediction: Typed prediction payload.
        analogs: Retrieved analog records.
        summary_statistics: Statistics derived from the analog set.
        requested_top_k: Requested analog count.

    Returns:
        Short deterministic grounded explanation text.
    """
    analog_count = int(summary_statistics["analog_count"])
    mean_distance = float(summary_statistics["mean_distance"])
    observed_event_rate = float(summary_statistics["observed_event_rate"])
    mean_duration_bars = float(summary_statistics["mean_duration_bars"])
    mean_boundary_span = float(summary_statistics["mean_boundary_span"])

    if prediction.low_confidence_advisory or analog_count < requested_top_k:
        return (
            f"Analog support is limited: the closest {analog_count} training analogs average distance "
            f"{mean_distance:.4f}, observed event rate {observed_event_rate:.2f}, and mean duration "
            f"{mean_duration_bars:.1f} bars. The current prediction remains advisory with boundary span "
            f"{mean_boundary_span:.4f}."
        )

    return (
        f"The top {analog_count} training analogs ground this prediction: they average distance "
        f"{mean_distance:.4f}, observed event rate {observed_event_rate:.2f}, mean duration "
        f"{mean_duration_bars:.1f} bars, and mean boundary span {mean_boundary_span:.4f}. "
        f"This supports the current event probability {prediction.event_probability:.2f}."
    )
