from enum import StrEnum


class ModelName(StrEnum):
    """Enumerates all model names in experiments."""

    IMAGE_JEPA_CONTEXT_ENCODER = "image_jepa_context_encoder"
    IMAGE_JEPA_TARGET_ENCODER = "image_jepa_target_encoder"
    IMAGE_JEPA_PREDICTOR = "jepa_predictor"

    TEMPORAL_ENCODER = "temporal_encoder"

    FORWARD_DYNAMICS = "forward_dynamics"

    POLICY_VALUE = "policy_value"
