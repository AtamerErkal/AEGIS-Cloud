# English: Core logic for Threat Evaluation using LangChain and Azure Functions
import logging
import json
import os
import azure.functions as func
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

def main(telemetryEvent: func.EventGridEvent, assessmentOutput: func.Out[func.Document]):
    logging.info('[AEGIS-CLOUD] Threat Evaluator triggered by Event Grid.')

    # 1. Get data from Jetson Nano via Event Grid
    try:
        raw_data = telemetryEvent.get_json()
        # Event Grid usually wraps IoT Hub data in the 'data' field
        payload = raw_data.get('data', raw_data) 
        logging.info(f"[DEBUG] Processing payload: {payload}")
    except Exception as e:
        logging.error(f"Error parsing event data: {e}")
        return

    # 2. Setup LangChain Intelligence
    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.2-chat-deployment", 
        api_version="2024-12-01-preview", 
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""
    SYSTEM: You are the AEGIS Strategic Defense AI. 
    Analyze the following drone detection data from an Edge device (Jetson Nano).
    
    DATA: {data}
    
    TASK: Provide a tactical assessment in JSON format:
    1. threat_score (0-100)
    2. action_recommendation (e.g., Jamming, Kinetic Strike, Observation)
    3. tactical_summary (Brief military-style explanation)
    """)

    # 3. Execute Reasoning
    chain = prompt | llm
    response = chain.invoke({"data": json.dumps(payload)})
    
    # Simple parsing logic (assuming LLM returns clean JSON or text)
    assessment_text = response.content
    logging.info(f"[AEGIS] Tactical Assessment: {assessment_text}")

    # 4. Save to Cosmos DB (Output Binding)
    # The 'assessmentOutput' binding automatically handles the write operation
    doc_id = f"assessment-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    output_doc = {
        "id": doc_id,
        "classification": "Tactical",
        "edge_device": "aegis-jetson-nano",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_assessment": assessment_text
    }
    
    assessmentOutput.set(func.Document.from_dict(output_doc))
    logging.info(f"[SUCCESS] Assessment {doc_id} final commit.")
    