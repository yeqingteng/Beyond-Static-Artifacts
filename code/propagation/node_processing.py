# -*- coding: utf-8 -*-
"""
This document is responsible for:
1) Processing logic for character nodes: determining interest and either rewriting or forwarding the information as is.
2) Audit logic for the Auditor: evaluating whether the current claim meets the client's requirements based on a single dimension.
"""
import json
import sys
import time
from typing import Dict, Any
from openai import OpenAI


MODEL = "gpt-4o"   
API_KEY = "xxxxxx"  
BASE_URL = "https://example.org"  

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

MAX_RETRIES = 3          
RETRY_BASE_DELAY = 2     


AUDITOR1_MODEL = "gpt-4o"
AUDITOR1_API_KEY = "xxxxxx"
AUDITOR1_BASE_URL = "https://example.org"

AUDITOR2_MODEL = "qwen3-max"
AUDITOR2_API_KEY = "xxxxxx"
AUDITOR2_BASE_URL = "https://example.org"

AUDITOR3_MODEL = "DeepSeek-V3.2"
AUDITOR3_API_KEY = "xxxxxx"
AUDITOR3_BASE_URL = "https://example.org"

auditor1_client = OpenAI(api_key=AUDITOR1_API_KEY, base_url=AUDITOR1_BASE_URL)
auditor2_client = OpenAI(api_key=AUDITOR2_API_KEY, base_url=AUDITOR2_BASE_URL)
auditor3_client = OpenAI(api_key=AUDITOR3_API_KEY, base_url=AUDITOR3_BASE_URL)


def _call_llm_generic(llm_client: OpenAI, model: str, prompt: str) -> str:
    
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            last_err = e
            wait_time = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"[call_llm_generic] Error: {e}. Retry in {wait_time}s...", file=sys.stderr)
            time.sleep(wait_time)

    print(f"[call_llm_generic] Failed after {MAX_RETRIES} retries. Last error: {last_err}", file=sys.stderr)
    return ""
# ========================================= #

def call_llm(prompt: str) -> str:
    return _call_llm_generic(client, MODEL, prompt)


def call_llm_auditor1(prompt: str) -> str:
    return _call_llm_generic(auditor1_client, AUDITOR1_MODEL, prompt)


def call_llm_auditor2(prompt: str) -> str:
    return _call_llm_generic(auditor2_client, AUDITOR2_MODEL, prompt)


def call_llm_auditor3(prompt: str) -> str:
    return _call_llm_generic(auditor3_client, AUDITOR3_MODEL, prompt)


# ========= Dimension description (used for constructing the review prompt)=========

DIMENSION_DESCRIPTIONS = {
    "linguistic_style": {
        "label": "Linguistic Style",
        "values": {
            "Rational–Official": "Formal, restrained, objective, and neutral in tone; resembles institutional statements, official documents, or professional reports; minimal emotional expression.",
            "Colloquial–Emotional": "Informal, conversational, and expressive; contains subjective feelings, attitudes, or emotional coloring, often using everyday or vivid language.",
            "any": "no specific requirement"
        }
    },
    "message_type": {
        "label": "Message Type",
        "values": {
            "Fact": "Presents objective, verifiable information or explanatory context without expressing subjective attitudes or evaluative judgments.",
            "Opinion": "Clearly and strongly expresses subjective attitudes, value judgments, personal interpretations, or evaluative viewpoints.",
            "Corrective": "From the speaker's own understanding, refute or correct facts or opinions that are inconsistent with the speaker's own knowledge.",
            "any": "no specific requirement"
        }
    },
    "granularity": {
        "label": "Granularity",
        "values": {
            "Macro/High-Level": "Focuses on overall trends, broader context, or general descriptions without detailed specifics.",
            "Micro/Detailed": "Includes concrete cases, examples, numbers, individuals, or fine-grained information.",
            "any": "no specific requirement"
        }
    },
    "causal_structure": {
        "label": "Causal Structure",
        "values": {
            "Direct Causation": "States that A directly causes B, with a clear and immediate causal link.",
            "Indirect Causation": "States that A influences B through intermediate factors or multi-step causal chains.",
            "Causal Refutation": "Rejects, disputes, or corrects a proposed causal relationship, asserting that “A does not / cannot / did not cause B.”",
            "No Causation": "Contains no causal relationship and does not address or refute causality.",
            "any": "no specific requirement"
        }
    },
    "stance_polarity": {
        "label": "Stance Polarity",
        "values": {
            "Support": "Clearly expresses agreement with a viewpoint, policy, action, person, or conclusion.",
            "Oppose": "Clearly expresses disagreement, criticism, or objection.",
            "Neutral": "Expresses no stance; presents information or viewpoints without favoring any side.",
            "Multi-perspective Comparison": "Presents and contrasts multiple positions or viewpoints without endorsing any single one.",
            "any": "no specific requirement"
        }
    }
}


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def process_character_node(
    character_id: str,
    attributes: Dict[str, Any],
    claim_text: str,
    dimension: str,
    target_value: str,
    base_claim: str,
    article_content: str = ""
) -> Dict[str, Any]:
    """
    Role node processing logic:
    - Determine if the role is suitable for [given dimension]. Rewrite the current claim as [target value];
    - If considered "unfit": act as a bystander and forward the current claim as is;
    - If considered "fit": act as a producer and focus on rewriting the claim in the target value direction for that dimension;
    - If the current claim information is insufficient, it can be supplemented with reference to article_content;
    - Additionally output character_reason:
    * List the top 3 attributes that influence whether it becomes a producer/bystander first
    * Then provide a brief explanation
    """

    if target_value == "any":
        return {
            "suitable": False,
            "role": "bystander",
            "updated_claim": claim_text,
            "top_attributes": [],
            "decision_reason": "",
            "character_reason": ""
        }

    
    attributes_str = json.dumps(attributes, ensure_ascii=False, indent=2)

    
    dim_cfg = DIMENSION_DESCRIPTIONS.get(dimension, {})
    dim_label = dim_cfg.get("label", dimension)
    human_value = dim_cfg.get("values", {}).get(target_value, target_value)

    article_content_str = article_content or ""

    prompt = f"""
You are a person in an information propagation network.
Your person ID is: {character_id}.

[Your attributes]
The following JSON describes your demographic and psychological attributes:
{attributes_str}

[Full article content related to this claim] (you may optionally use this):
\"\"\"{article_content_str}\"\"\"

[Current claim you received in this step]
\"\"\"{claim_text}\"\"\"

[Client requirement for this step]
You are working on EXACTLY ONE customization dimension:

- Dimension (internal name): {dimension}
- Human-readable label: {dim_label}
- Target value (human-readable): {human_value}
- Target value (internal): {target_value}

Your tasks:

1. VERY IMPORTANT: Keep in mind that you are simulating a real person in this role-play. 
   Based on your attributes, decide whether you are a suitable person to rewrite the current claim
   so that it better satisfies the client's requirement on THIS dimension (and this dimension only)

   - If you think you are NOT suitable for this dimension and target value,
     you act as a "bystander":
       * You do NOT change any wording of the current claim;
       * You simply forward it unchanged.

   - If you think you ARE suitable for this dimension and target value,
     you act as a "producer":
       * Keep in mind that you are simulating a real person in this role-play.
       You rewrite the CURRENT claim, focusing on adjusting it along this dimension toward the target value described above in your own way
       (The statement must not be rewritten as an interrogative sentence and must not end with a question mark!).
       * If you judge yourself suitable but the claim does not contain enough information to express the required target value, 
       you MAY refer to the corresponding article content SOLELY to supplement the missing elements needed for the rewrite.
       * DO NOT expand the content arbitrarily! The length of the rewritten claim MUST NOT differ significantly from the length of the original claim!

2. VERY IMPORTANT: Explain WHY you act as a producer or bystander.
   - Identify the TOP 3 attributes from your attribute JSON that most strongly influence your decision
     to be a producer or a bystander in this step.
   - Give a brief explanation (1–2 sentences) about how these attributes lead to your role decision
     and your suitability (or lack of suitability) for this dimension and target value.

Output requirements:
- ONLY output a JSON object, no extra text;
- Use lowercase true/false for booleans;
- The JSON MUST have exactly the following fields:
  - "suitable": boolean, whether you think you are suitable to perform the rewrite on this dimension;
  - "role": string, one of "producer" or "bystander";
  - "updated_claim": string, the final text you forward in this step;
  - "top_attributes": array of strings, listing 3 attribute keys;
  - "reason": string, a brief explanation (1–2 sentences) describing why these attributes matter for your decision.

Example output:
{{
  "suitable": true,
  "role": "producer",
  "updated_claim": "Here is your rewritten claim...",
  "top_attributes": ["political_leaning", "education", "big_five"],
  "reason": "Because I am highly educated and strongly support this policy, I am motivated and able to rewrite the claim in my way that......"
}}
"""
    raw = call_llm(prompt)
    data = _safe_json_loads(raw)

    # Conservative strategy when parsing fails: Treat as "inapplicable" and simply forward as is
    suitable = bool(data.get("suitable", data.get("interest", False)))
    role = data.get("role", "bystander")
    updated_claim = data.get("updated_claim", claim_text)

    # Process top_attributes (selecting at most the first three)
    raw_top_attrs = data.get("top_attributes", [])
    if isinstance(raw_top_attrs, str):
        top_attributes = [x.strip() for x in raw_top_attrs.split(",") if x.strip()]
    elif isinstance(raw_top_attrs, list):
        top_attributes = [str(x).strip() for x in raw_top_attrs if str(x).strip()]
    else:
        top_attributes = []
    if len(top_attributes) > 3:
        top_attributes = top_attributes[:3]

    decision_reason = str(data.get("reason", "")).strip()

    # If the model deems itself unsuitable, it will be forced to act as a bystander and simply forward the message as it is.
    if not suitable:
        role = "bystander"
        updated_claim = claim_text

    if top_attributes and decision_reason:
        character_reason = (
            f"Top attributes influencing role ({role}): "
            + ", ".join(top_attributes)
            + f". Explanation: {decision_reason}"
        )
    elif top_attributes:
        character_reason = (
            f"Top attributes influencing role ({role}): " + ", ".join(top_attributes)
        )
    elif decision_reason:
        character_reason = decision_reason
    else:
        character_reason = f"Role decided as {role} based on character attributes."

    return {
        "suitable": suitable,
        "role": role,
        "updated_claim": updated_claim,
        "top_attributes": top_attributes,
        "decision_reason": decision_reason,
        #"character_reason": character_reason
        "influencing_attributes": top_attributes,
        "influencing_reason": decision_reason,
    }


def audit_dimension(
    claim_text: str,
    dimension: str,
    target_value: str,
    auditor_id: str = "auditor1",
) -> Dict[str, Any]:
    
    
    if target_value == "any":
        return {"pass": True, "reason": ""}

    dim_cfg = DIMENSION_DESCRIPTIONS.get(dimension, None)
    if dim_cfg is None:
        return {"pass": True, "reason": f"Unknown dimension '{dimension}', treated as passed by default."}

    dim_label = dim_cfg["label"]
    human_value = dim_cfg["values"].get(target_value, target_value)

    if auditor_id == "auditor2":
        _call = call_llm_auditor2
    elif auditor_id == "auditor3":
        _call = call_llm_auditor3
    else:
        _call = call_llm_auditor1

    prompt = f"""
You are an auditor (id = {auditor_id}) who checks whether a message satisfies a client's requirement on exactly ONE specific dimension.

[Message to evaluate]
\"\"\"{claim_text}\"\"\"

[Target dimension]
- Dimension (internal name): {dimension}
- Human-readable label: {dim_label}
- Target value (human-readable): {human_value}
- Target value (internal): {target_value}

Your job:
- Focus ONLY on this single dimension when making your decision.
- If the target dimension's characteristics are highly salient, clearly prominent, strongly expressed, or represent a typical example, the message should be marked as passed (pass = true).
- If the target dimension's characteristics are not sufficiently salient, not prominent, expressed ambiguously, or only partially aligned, the message should be marked as not passed (pass = false).

Output requirements:
- ONLY output a JSON object, no extra text;
- Use lowercase true/false for booleans;
- The JSON MUST have exactly the following fields:
  - "pass": boolean
  - "reason": string, a brief explanation (one or two sentences).

Example output:
{{
  "pass": true,
  "reason": "The tone is formal and neutral, which matches the rational and official style."
}}
"""

    raw = _call(prompt)
    data = _safe_json_loads(raw)

    passed = bool(data.get("pass", False))
    reason = data.get("reason", "")

    return {
        "pass": passed,
        "reason": reason
    }


__all__ = [
    "call_llm",
    "call_llm_auditor1",
    "call_llm_auditor2",
    "call_llm_auditor3",
    "process_character_node",
    "audit_dimension",
    "DIMENSION_DESCRIPTIONS",
    "MODEL",
    "API_KEY",
    "BASE_URL"
]

