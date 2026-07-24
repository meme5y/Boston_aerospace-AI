"""
Boston Aerospace AI
OpenAI Intelligence Orchestrator

OpenAI-only architecture.

Responsibilities:
- Engineering chat
- RUL interpretation
- SHAP interpretation
- RAG context
- Optional web research
- Structured engineering responses
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from Config.Settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


class AIOrchestrator:

    def __init__(self):

        if not OPENAI_API_KEY:

            raise RuntimeError(
                "OPENAI_API_KEY não configurada."
            )

        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

        self.model = OPENAI_MODEL

    # ========================================================
    # CHAT PRINCIPAL
    # ========================================================

    def chat(
        self,
        message: str,
        context: Optional[str] = None,
        use_web: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:

        system_prompt = """
You are Boston Aerospace AI.

You are an aerospace engineering AI assistant specialized in:

- Aircraft engine predictive maintenance
- Remaining Useful Life (RUL)
- NASA CMAPSS
- Turbofan engines
- PHM
- SHAP explainability
- Sensor degradation
- Engine health monitoring
- Maintenance engineering
- Aerospace technical documentation

Your answers must be:

1. Technically accurate
2. Structured
3. Clear
4. Engineering-focused
5. Honest about uncertainty

When RUL or ML results are provided, explain:

- What the prediction means
- Which sensors contributed
- Confidence and uncertainty
- Possible failure mechanisms
- Recommended maintenance action

Do not invent sensor values or measurements.
"""

        user_prompt = message

        if context:

            user_prompt += f"""

CONTEXT FROM BOSTON AEROSPACE AI:

{context}
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        if history:

            messages.extend(
                history[-10:]
            )

        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        kwargs = {
            "model": self.model,
            "input": messages,
        }

        if use_web:

            kwargs["tools"] = [
                {
                    "type": "web_search_preview"
                }
            ]

        response = self.client.responses.create(
            **kwargs
        )

        text = response.output_text

        return {
            "success": True,
            "answer": text,
            "model": self.model,
            "provider": "openai",
            "web_used": use_web,
        }

    # ========================================================
    # ANÁLISE DE RUL
    # ========================================================

    def analyze_rul(
        self,
        rul_data: Dict[str, Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:

        prompt = f"""
Analyze this aerospace engine RUL prediction.

PREDICTION DATA:

{rul_data}

Provide:

1. Executive interpretation
2. Risk level
3. Important sensor contributions
4. Possible degradation mechanisms
5. Maintenance recommendation
6. Uncertainty interpretation
"""

        return self.chat(
            message=prompt,
            context=context,
        )

    # ========================================================
    # EXPLAIN SHAP
    # ========================================================

    def explain_shap(
        self,
        shap_data: List[Dict[str, Any]],
        rul: Optional[float] = None,
    ) -> Dict[str, Any]:

        prompt = f"""
Interpret these SHAP feature contributions from an aerospace
engine predictive maintenance model.

SHAP FEATURES:

{shap_data}

RUL:

{rul}

Explain:

- Most important sensors
- Direction of their influence
- Possible physical interpretation
- Recommended engineering investigation
"""

        return self.chat(
            message=prompt
        )


# ============================================================
# SINGLETON
# ============================================================

_orchestrator = None


def get_orchestrator() -> AIOrchestrator:

    global _orchestrator

    if _orchestrator is None:

        _orchestrator = AIOrchestrator()

    return _orchestrator


def ask_ai(
    message: str,
    context: Optional[str] = None,
    use_web: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:

    orchestrator = get_orchestrator()

    return orchestrator.chat(
        message=message,
        context=context,
        use_web=use_web,
        history=history,
        )
