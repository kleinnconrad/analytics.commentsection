"""Versioned, descriptive thresholds for the exploratory analysis."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Configure descriptive flags, not calibrated probabilities or confidence levels."""

    min_posts: int = 5
    temporal_posts_per_day: int = 10
    business_hours_share: float = 0.75
    response_minutes: int = 60
    min_topic_posts: int = 10
    max_topic_entropy: float = 0.2
    duplicate_share: float = 0.5
    impact_quantile: float = 0.9
    min_political_posts: int = 10
    political_majority: float = 0.68

    def __post_init__(self) -> None:
        """Reject thresholds that cannot express the documented rules."""
        for name in (
            "min_posts",
            "min_topic_posts",
            "min_political_posts",
            "temporal_posts_per_day",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("business_hours_share", "duplicate_share", "impact_quantile"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0.5 < self.political_majority <= 1:
            raise ValueError("political_majority must be in (0.5, 1]")
        if self.response_minutes < 0 or self.max_topic_entropy < 0:
            raise ValueError("Response time and entropy thresholds must be nonnegative")


FLAGS = ["flag_temporal", "flag_topic", "flag_repetition", "flag_impact"]
