"""U10 degraded-mode and low-confidence advisory logic."""

from __future__ import annotations

from dataclasses import dataclass

from models.explanation import RetrievalEvidence


class AdvisoryError(ValueError):
    """Raised when advisory evaluation fails."""


@dataclass(frozen=True)
class AdvisoryDecision:
    """Typed degraded-mode advisory decision."""

    low_confidence_advisory: bool
    reasons: tuple[str, ...]


def evaluate_advisory(
    *,
    confidence: float,
    evidence: RetrievalEvidence,
    requested_top_k: int,
) -> AdvisoryDecision:
    """Determine whether inference should emit a low-confidence advisory."""
    if not 0.0 <= confidence <= 1.0:
        raise AdvisoryError("confidence must lie in [0, 1].")
    if requested_top_k <= 0:
        raise AdvisoryError("requested_top_k must be positive.")

    reasons: list[str] = []
    if confidence < 0.5:
        reasons.append("low_model_confidence")
    if len(evidence.analogs) < requested_top_k:
        reasons.append("weak_analog_support")

    return AdvisoryDecision(
        low_confidence_advisory=bool(reasons),
        reasons=tuple(reasons),
    )
