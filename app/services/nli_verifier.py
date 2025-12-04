from app.core.models import ModelManager
from app.core.exceptions import NLIVerificationException


def nli_score(claim: str, snippet: str, use_context: bool = None) -> float:
    """
    Calculate NLI (Natural Language Inference) entailment score.

    Uses roberta-large-mnli to determine if the snippet entails the claim.

    Args:
        claim: The hypothesis/claim to verify
        snippet: The premise/evidence text
        use_context: Whether to add "Established fact:" prefix (None = use settings)

    Returns:
        Float between 0.0 and 1.0 representing entailment probability

    Raises:
        NLIVerificationException: If NLI verification fails
    """
    try:
        # Get NLI pipeline
        mm = ModelManager.get_instance()
        nli = mm.get_nli()

        # Get feature flag from settings if not explicitly provided
        from app.core.config import settings
        if use_context is None:
            use_context = settings.use_nli_context

        # Format: premise </s></s> hypothesis
        # This is the format roberta-large-mnli expects
        # Optional contextual prefix to guide model toward recognizing established facts
        if use_context:
            hypothesis = f"Established fact: {claim}"
        else:
            hypothesis = claim

        premise = snippet
        input_text = f"{premise} </s></s> {hypothesis}"

        # Run NLI pipeline
        result = nli(input_text)

        # Extract entailment score
        # Result is a list of dicts with 'label' and 'score'
        if isinstance(result, list):
            ent_score = None
            for r in result:
                label = r['label'].upper()
                if 'ENTAIL' in label:
                    ent_score = r['score']
                    break

            # If no entailment label found, return 0.0
            if ent_score is None:
                ent_score = 0.0

            return float(ent_score)
        else:
            # Fallback
            return 0.0

    except Exception as e:
        raise NLIVerificationException(
            f"NLI verification failed: {str(e)}",
            details={"claim": claim[:50], "evidence": snippet[:50], "error": str(e)}
        )
