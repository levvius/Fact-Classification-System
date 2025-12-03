from typing import Dict, List, Optional
import logging

from app.services.claim_extractor import extract_claims
from app.services.evidence_retriever import retrieve_proofs
from app.services.nli_verifier import nli_score
from app.core.config import settings
from app.core.exceptions import ClassificationException

logger = logging.getLogger(__name__)


def assess_claim(claim: str, top_k: int = None) -> Dict:
    """
    Assess a single claim by retrieving evidence and computing NLI scores.

    Args:
        claim: The claim text to assess
        top_k: Number of evidence snippets to retrieve

    Returns:
        Dict with keys: claim, support, best_proof, all_proofs
    """
    if top_k is None:
        top_k = settings.top_k_proofs

    # Retrieve evidence
    proofs = retrieve_proofs(claim, top_k=top_k)

    # Calculate NLI scores for each proof
    scores = []
    best_proof = None
    best_score = -1.0

    for proof in proofs:
        nli_ent_score = nli_score(claim, proof["snippet"])
        proof["nli_score"] = nli_ent_score
        scores.append(nli_ent_score)

        if nli_ent_score > best_score:
            best_score = nli_ent_score
            best_proof = proof

    # Aggregate: use max entailment as support score
    support = max(scores) if scores else 0.0

    return {
        "claim": claim,
        "support": support,
        "best_proof": best_proof,
        "all_proofs": proofs
    }


def classify_text(text: str) -> Dict:
    """
    Classify text as "правда", "неправда", or "нейтрально".

    Args:
        text: Input text to classify

    Returns:
        Dict with keys:
            - overall_classification: str ("правда", "неправда", "нейтрально")
            - confidence: float
            - claims: List[Dict] with claim analysis

    Raises:
        ClassificationException: If classification fails
    """
    import sys
    try:
        # Extract claims
        logger.info("STEP 1: Starting claim extraction...")
        sys.stdout.flush()

        claims = extract_claims(text)
        logger.info(f"STEP 1: ✓ Extracted {len(claims)} claims")
        sys.stdout.flush()

        # Limit claims if needed
        if len(claims) > settings.max_claims:
            logger.info(f"STEP 2: Limiting to {settings.max_claims} claims")
            claims = claims[:settings.max_claims]
            sys.stdout.flush()

        # Assess each claim
        logger.info("STEP 3: Starting claim assessment...")
        sys.stdout.flush()

        claim_results = []
        for i, claim_text in enumerate(claims, 1):
            logger.info(f"STEP 3.{i}: Assessing claim: {claim_text[:50]}...")
            sys.stdout.flush()
            result = assess_claim(claim_text)

            # Map support score to classification
            support = result["support"]
            if support >= settings.truth_threshold:
                classification = "правда"
                confidence = support
            elif support < settings.falsehood_threshold:
                classification = "неправда"
                confidence = 1.0 - support
            else:
                classification = "нейтрально"
                confidence = support

            claim_results.append({
                "claim": claim_text,
                "classification": classification,
                "confidence": confidence,
                "best_evidence": {
                    "snippet": result["best_proof"]["snippet"],
                    "source": result["best_proof"]["source"],
                    "nli_score": result["best_proof"]["nli_score"],
                    "retrieval_score": result["best_proof"]["retrieval_score"]
                } if result["best_proof"] else None
            })

            logger.info(f"STEP 3.{i}: ✓ {classification} (support: {support:.2f})")
            sys.stdout.flush()

        # Overall classification aggregation
        logger.info("STEP 4: Aggregating results...")
        sys.stdout.flush()
        # Priority: if any "неправда" -> overall "неправда"
        #           elif any "нейтрально" -> overall "нейтрально"
        #           else -> overall "правда"
        classifications = [r["classification"] for r in claim_results]
        confidences = [r["confidence"] for r in claim_results]

        if "неправда" in classifications:
            overall = "неправда"
            # Average confidence of falsehood claims
            falsehood_confidences = [
                c for c, cl in zip(confidences, classifications) if cl == "неправда"
            ]
            overall_confidence = sum(falsehood_confidences) / len(falsehood_confidences)
        elif "нейтрально" in classifications:
            overall = "нейтрально"
            # Average confidence of neutral claims
            neutral_confidences = [
                c for c, cl in zip(confidences, classifications) if cl == "нейтрально"
            ]
            overall_confidence = sum(neutral_confidences) / len(neutral_confidences)
        else:
            overall = "правда"
            # Average confidence of truth claims
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(f"STEP 4: ✓ Overall: {overall} (confidence: {overall_confidence:.2f})")
        sys.stdout.flush()

        return {
            "overall_classification": overall,
            "confidence": overall_confidence,
            "claims": claim_results
        }

    except Exception as e:
        raise ClassificationException(
            f"Classification failed: {str(e)}",
            details={"text_length": len(text), "error": str(e)}
        )
