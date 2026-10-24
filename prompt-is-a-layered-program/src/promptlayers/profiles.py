"""What each model accepts, per layer. The only place provider differences live."""

from __future__ import annotations

from dataclasses import dataclass, field

# Keywords some providers reject (Anthropic: 400) or silently ignore (Gemini).
NON_PORTABLE_KEYWORDS = frozenset({"minimum", "maximum", "minLength", "maxLength"})


@dataclass(frozen=True)
class ModelProfile:
    model: str  # LiteLLM model id, e.g. "openai/gpt-4o-2024-08-06"
    supports_json_schema: bool = True
    supports_temperature: bool = True
    rejected_schema_keywords: frozenset[str] = field(default=frozenset())
    instruction_override: str | None = None  # L1 rewritten for this model


def profile_from_litellm(model: str, **overrides: object) -> ModelProfile:
    """Build a profile from LiteLLM's bundled model map instead of hand-written flags.

    The map is a starting point, not the provider's word; pass overrides where they differ.
    """
    import litellm

    params = litellm.get_supported_openai_params(model=model) or []
    detected = {
        "supports_json_schema": litellm.supports_response_schema(model=model),
        "supports_temperature": "temperature" in params,
        # Send only the portable schema subset everywhere; Pydantic enforces the rest.
        "rejected_schema_keywords": NON_PORTABLE_KEYWORDS,
    }
    return ModelProfile(model=model, **{**detected, **overrides})  # type: ignore[arg-type]
