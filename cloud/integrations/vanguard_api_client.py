"""
AEGIS-Cloud — Cloud / Integrations / VANGUARD API Client
==========================================================
Module:   cloud.integrations.vanguard_api_client
Platform: Azure (serverless / container)

PURPOSE
-------
Modular connector to the **VANGUARD** Threat Classification Model,
which is maintained as a SEPARATE project with its own MLOps lifecycle.

AEGIS-Cloud treats VANGUARD as an *external Intelligence API* — this
client is the ONLY integration point.  No VANGUARD model weights,
training code, or internal schemas are duplicated inside AEGIS-Cloud.

This strict separation demonstrates **multi-project lifecycle
management**: VANGUARD can be retrained, versioned, and deployed
independently; AEGIS-Cloud only needs to know the API contract.

API CONTRACT (Expected)
-----------------------
  POST /api/v1/classify
  Request:
    {
      "image_b64":       str,   // Base64-encoded detection crop
      "detection_meta":  dict,  // YOLO bbox, confidence, class_id
      "context":         str    // Optional RAG-provided doctrinal context
    }
  Response:
    {
      "classification":  str,   // "Friendly" | "Unknown" | "Hostile"
      "confidence":      float,
      "shap_values":     dict,  // SHAP attribution for the classification
      "grad_cam_b64":    str,   // Grad-CAM heatmap overlay (base64 PNG)
      "model_version":   str    // Semantic version of the VANGUARD model
    }

DESIGN PRINCIPLES
-----------------
1. **Project Separation**
   This file is the SOLE bridge to VANGUARD.  All VANGUARD-specific
   logic (endpoint URLs, auth tokens, payload schemas) is encapsulated
   here.

2. **EU AI Act — XAI Mandate**
   The client MUST request and receive SHAP values and Grad-CAM
   evidence with every classification response.  If the VANGUARD API
   fails to return XAI data, the classification is rejected and logged
   as an audit anomaly.

3. **AIOps — Resilience**
   - Implements exponential back-off retry (configurable).
   - Publishes API latency and error-rate metrics to ``data/logs/``.
   - Falls back to a "VANGUARD_UNAVAILABLE" status (does NOT guess
     a classification).

4. **NATO-Standard Logging**
   Every API call and response is logged in the NATO Incident Report
   format, including the ``model_version`` field for traceability.

INTERFACES
----------
- Consumed by: ``ThreatEvaluatorAgent``
- Configuration: Environment variables (``VANGUARD_ENDPOINT``,
  ``VANGUARD_API_KEY``).

SPRINT ASSIGNMENT
-----------------
Day 2:   Define client class, request/response schemas, auth flow.
Day 3:   Implement retry logic and XAI validation guard.
Day 4:   Integration test with a mock VANGUARD endpoint.
"""


class VanguardAPIClient:
    """
    HTTP client for the VANGUARD Threat Classification API.

    Attributes
    ----------
    endpoint : str
        Base URL of the VANGUARD API (from env ``VANGUARD_ENDPOINT``).
    api_key : str
        Bearer token for request authentication.
    timeout_seconds : int
        Per-request timeout (default: 10).
    max_retries : int
        Exponential back-off retry limit (default: 3).
    """

    def __init__(self):
        """Load endpoint URL and API key from environment variables."""
        ...

    def classify(self, image_b64: str, detection_meta: dict,
                 context: str = "") -> dict:
        """
        Submit a detection crop to VANGUARD for threat classification.

        Returns the full classification response including XAI evidence.
        Raises ``VanguardXAIMissingError`` if SHAP/Grad-CAM data is
        absent from the response.
        """
        ...

    def _validate_xai_response(self, response: dict) -> bool:
        """
        Enforce EU AI Act XAI mandate: reject any response missing
        ``shap_values`` or ``grad_cam_b64``.
        """
        ...

    def health_check(self) -> dict:
        """Ping the VANGUARD /health endpoint and return status."""
        ...


class VanguardXAIMissingError(Exception):
    """Raised when VANGUARD response lacks mandatory XAI evidence."""
    ...
