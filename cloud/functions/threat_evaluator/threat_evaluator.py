"""
AEGIS-Cloud — Cloud / Functions / Threat Evaluator
====================================================
Azure Function triggered by IoT Hub via Event Grid.

Pipeline:
  1. Parse NATO STANAG-4586 payload from Jetson Nano.
  2. Extract Moondream VLM reports as XAI evidence.
  3. Run GPT-5.2 tactical assessment via LangChain.
  4. Parse structured JSON response from LLM.
  5. Persist full audit record to Cosmos DB.
"""

import json
import logging
import os
import re
from datetime import datetime

try:
    import azure.functions as func
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import AzureChatOpenAI
    _AZURE_AVAILABLE = True
except ImportError:
    _AZURE_AVAILABLE = False


def main(telemetryEvent, assessmentOutput):
    logging.info("[AEGIS-CLOUD] Threat Evaluator triggered by Event Grid.")

    # ------------------------------------------------------------------
    # 1. Parse incoming NATO payload from Jetson Nano
    # ------------------------------------------------------------------
    try:
        raw_data = telemetryEvent.get_json()
        payload = raw_data.get("data", raw_data)
        logging.info(f"[AEGIS] Processing message_id={payload.get('message_id')}")
    except Exception as e:
        logging.error(f"[AEGIS] Error parsing event data: {e}")
        return

    # ------------------------------------------------------------------
    # 2. Extract Moondream VLM reports as XAI evidence
    # ------------------------------------------------------------------
    reasoning_reports = payload.get("reasoning", [])
    xai_evidence = [
        r.get("description", "")
        for r in reasoning_reports
        if r.get("status") == "OK" and r.get("description")
    ]
    xai_text = "\n".join(xai_evidence) if xai_evidence else "No VLM report available."

    # ------------------------------------------------------------------
    # 3. LangChain GPT-5.2 tactical assessment
    # ------------------------------------------------------------------
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.2-chat-deployment",
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_template(
        """SYSTEM: You are the AEGIS Strategic Defense AI.
Analyze the following drone detection data from an Edge device (Jetson Nano).

DETECTION DATA:
{detections}

EDGE XAI EVIDENCE (Moondream VLM tactical reports):
{xai}

TASK: Respond ONLY with valid JSON — no markdown, no extra text:
{{
    "threat_score": <int 0-100>,
    "action_recommendation": "<Jamming | Kinetic Strike | Observation | Safe>",
    "tactical_summary": "<Brief military-style explanation, max 2 sentences>",
    "confidence": <float 0.0-1.0>
}}"""
    )

    chain = prompt | llm
    response = chain.invoke({
        "detections": json.dumps(payload.get("detections", []), indent=2),
        "xai": xai_text,
    })

    # ------------------------------------------------------------------
    # 4. Parse structured JSON from LLM response
    # ------------------------------------------------------------------
    assessment = _parse_llm_json(response.content)
    logging.info(f"[AEGIS] Tactical Assessment: {assessment}")

    # ------------------------------------------------------------------
    # 5. Persist full audit record to Cosmos DB
    # ------------------------------------------------------------------
    nato_meta = payload.get("nato_metadata", {})
    hints = payload.get("azure_function_hints", {})
    doc_id = f"assessment-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

    output_doc = {
        "id": doc_id,
        "schema_version": "1.1",
        "classification": "Tactical",
        "edge_device": nato_meta.get("station_id", "aegis-jetson-nano"),
        "timestamp_utc": datetime.utcnow().isoformat(),
        "message_id": payload.get("message_id"),
        # LLM assessment
        "threat_score": assessment.get("threat_score"),
        "action_recommendation": assessment.get("action_recommendation"),
        "tactical_summary": assessment.get("tactical_summary"),
        "confidence": assessment.get("confidence"),
        # XAI evidence (Moondream verbatim — EU AI Act Article 13)
        "xai_evidence": xai_evidence,
        # Human-in-the-Loop gate (EU AI Act Article 14)
        "human_review_required": hints.get("human_review_required", False),
        "human_decision": None,  # Populated by operator dashboard
        # Raw data for audit trail
        "raw_detections": payload.get("detections", []),
        "hardware_telemetry": payload.get("hardware_telemetry", {}),
        "nato_metadata": nato_meta,
    }

    assessmentOutput.set(func.Document.from_dict(output_doc))
    logging.info(f"[SUCCESS] Assessment {doc_id} saved to Cosmos DB.")


def _parse_llm_json(text: str) -> dict:
    """Extract and parse JSON from LLM response with fallback."""
    clean = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logging.warning("[AEGIS] LLM response was not valid JSON. Storing raw text.")
    return {"raw_response": text, "parse_error": True}
