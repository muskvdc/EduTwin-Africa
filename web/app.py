from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

from week1_llm.pipeline import AssistantPipeline
from week1_llm.security.budget import BudgetExceeded
from week1_llm.security.exceptions import SecurityBlocked


PROJECT_ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="EduTwin Africa — Auto Agent Runtime")
pipeline = AssistantPipeline(knowledge_dir=PROJECT_ROOT / "data" / "knowledge")
INDEX = Path(__file__).parent / "static" / "index.html"
LOGGER = logging.getLogger("edutwin_africa")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    temperature: float = Field(default=0.7, ge=0, le=2)
    mode: Literal["auto", "normal", "multi_agent"] = "normal"


@app.get("/")
def home():
    return FileResponse(INDEX)


@app.get("/api/security/status")
def security_status():
    return {
        "input_guardrails": True,
        "output_guardrails": True,
        "prompt_injection_detection": True,
        "secret_detection": True,
        "pii_auditing": True,
        "rate_limiting": True,
        "request_budgets": True,
        "tool_allowlist": True,
        "tool_approval_gates": True,
        "red_team_suite": True,
    }


@app.post("/api/chat")
def chat(request: ChatRequest, http_request: Request = None):
    security = getattr(pipeline, "security", None)
    request_id = security.new_request_id() if security is not None else "test-request"
    actor_id = "local"
    if http_request is not None and http_request.client is not None:
        actor_id = http_request.client.host or "local"

    try:
        if security is not None:
            security.check_rate_limit(actor_id, request_id)
        if request.mode == "auto":
            if security is None:
                result = pipeline.auto_chat(request.message, request.temperature)
            else:
                result = pipeline.auto_chat(
                    request.message,
                    request.temperature,
                    request_id=request_id,
                    actor_id=actor_id,
                )
            return {"mode": "auto", "request_id": request_id, **result}

        if request.mode == "multi_agent":
            if security is None:
                result = pipeline.multi_agent_chat(request.message, request.temperature)
            else:
                result = pipeline.multi_agent_chat(
                    request.message,
                    request.temperature,
                    request_id=request_id,
                    actor_id=actor_id,
                )
            return {"mode": "multi_agent", "request_id": request_id, **result}

        if security is None:
            answer = pipeline.chat(request.message, request.temperature)
        else:
            answer = pipeline.chat(
                request.message,
                request.temperature,
                request_id=request_id,
                actor_id=actor_id,
            )
        return {"mode": "normal", "request_id": request_id, "answer": answer, "trace": []}

    except SecurityBlocked as exc:
        raise HTTPException(status_code=400, detail=exc.decision.user_message) from exc
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        if security is not None:
            security.audit.log(
                "unhandled_request_error",
                request_id=request_id,
                actor_id=actor_id,
                severity="error",
                details={"error_type": type(exc).__name__},
            )
        LOGGER.exception("Request %s failed", request_id)
        raise HTTPException(
            status_code=500,
            detail="The request could not be completed. Check the server log using the request ID.",
        ) from exc
