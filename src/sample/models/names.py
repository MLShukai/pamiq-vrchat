from enum import StrEnum


class ModelName(StrEnum):
    """Enumerates all model names in experiments."""

    JEPA_CONTEXT_ENCODER = "jepa_context_encoder"
    JEPA_TARGET_ENCODER = "jepa_target_encoder"
    JEPA_PREDICTOR = "jepa_predictor"

    FORWARD_DYNAMICS = "forward_dynamics"
