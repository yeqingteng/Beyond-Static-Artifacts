from typing import Dict, Any, List, Optional, Tuple
import json
import re

def _format_character_profile(character_profile: Dict[str, Any]) -> str:
    ordered_keys = [
        "name",
        "religious",
        "employment",
        "marital",
        "ideology",
        "income",
        "area",
        "age",
        "gender",
        "big_five",
        "education",
        "job"
    ]

    lines = []
    for key in ordered_keys:
        if key in character_profile:
            lines.append(f"- {key}: {character_profile[key]}")

    for key, value in character_profile.items():
        if key not in ordered_keys:
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


# ==============  Hub's viewpoint generation  ==============

def build_hub_opinion_messages(
    hub_id: str,
    hub_profile: Dict[str, Any],
    current_claim: str,
) -> List[Dict[str, str]]:
    """
    Generate messages that provide a simple view of the hub node regarding the "current_claim" (1-2 sentences). 
    Output format: JSON only.
    {
      "hub_opinion": "..."   
    }
    """
    profile_text = _format_character_profile(hub_profile)

    system_prompt = f"""
    You are a highly influential opinion leader (hub) in an online social network.
    You have the following personal profile:
    {profile_text}

    You will read a CURRENT_CLAIM and then produce a short opinion about it
    in *one or two sentences*.

    Important:
    - Your opinion should sound like a realistic social media post or public comment
      that someone with your profile might make.
    - The tone, stance, and style should be consistent with your profile.
    - You must respond ONLY in strict JSON with exactly one field:
        "hub_opinion": "<one or two sentences here>"

    Example:
    {{
      "hub_opinion": "......"
    }}
    """

    user_prompt = f"""
    CURRENT_CLAIM:
    {current_claim}

    Please give your short opinion about this claim (1-2 sentences) in the required JSON format.
    """

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_hub_opinion_response(raw_output: str) -> str:
    
    text = raw_output.strip()
    text = _extract_json_from_text(text)

    try:
        data = json.loads(text)
        opinion = data.get("hub_opinion") or ""
        return opinion.strip()
    except Exception:
        return raw_output.strip()


# ==============  Character Rewrite + FUSE-EVAL  ==============

def build_character_messages(
    character_id: str,
    character_profile: Dict[str, Any],
    base_claim: str,
    current_claim: Optional[str] = None,
    influencing_hub_id: Optional[str] = None,
    influencing_hub_opinion: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Construct the list of messages passed to the Character (Producer)
    - character_profile: Attributes of the current node
    - base_claim: The original input claim (unchanged)
    - current_claim: The claim from the previous step. If it's the first step, it can be directly used as base_claim
    - influencing_hub_opinion: In this step, the hub node provides 1-2 viewpoints regarding the current_claim.
    The current character will be influenced by these viewpoints during rewriting.
    """
    profile_text = _format_character_profile(character_profile)

    system_prompt = f"""
    You are a claim consumer with the specific personal profile described below.
    Your task is to rewrite or adjust the given claim in a way that reflects your personal profile,
    attitudes, perspective, and other relevant factors.

    Very important:
    - You must respond in STRICT JSON format only, with no extra text.
    - The JSON must have exactly these two fields:
        "rewritten_claim": a single rewritten claim sentence (string),
        "top_attributes": an array of EXACTLY three attribute names (strings) chosen
          from your own profile keys
          ("age", "gender", "religious", "ideology", "area", "income", "education",
           "job", "marital", "big_five")
          that MOST influenced how you rewrote the claim.

    Example of the expected JSON format:
    {{
      "rewritten_claim": ".....",
      "top_attributes": ["age", "gender", "religious"]
    }}

    Here is your profile:
    {profile_text}
    """

    if influencing_hub_id is not None and influencing_hub_opinion:
        system_prompt += f"""

    In this step, you are also influenced by an important hub character in the network.

    HUB_ID: {influencing_hub_id}
    HUB_OPINION_ABOUT_CURRENT_CLAIM:
    "{influencing_hub_opinion}"

    You have recently seen this hub's public opinion on the CURRENT version of the claim.
    When you rewrite the claim, you should consider BOTH:
    - your own personal profile, and
    - the influence of this hub's opinion (stance, tone, etc.)
    """

    if current_claim is None:
        current_claim = base_claim

    user_prompt = f"""
    Here is the current version of a claim circulating in the network:
    {current_claim}

    Keep in mind that you are simulating a real person.
    Please generate a new version of the claim that reflects:
    - your own personal profile, attitudes, perspective, and other relevant factors; and
    - the influence of any hub opinion described in the system message (if present).

    Again, output ONLY the rewritten claim sentence in the JSON field "rewritten_claim",
    plus the three most influential profile attributes in "top_attributes".
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def parse_character_response(raw_output: str) -> Tuple[str, str]:
    
    text = raw_output.strip()
    text = _extract_json_from_text(text)

    try:
        data = json.loads(text)
        rewritten = (data.get("rewritten_claim") or "").strip()

        top_attrs = data.get("top_attributes") or []
        if isinstance(top_attrs, list):
            attr_str = "; ".join(str(a) for a in top_attrs[:3])
        else:
            attr_str = str(top_attrs) if top_attrs is not None else ""
    except Exception:
        rewritten = text
        attr_str = ""

    return rewritten, attr_str


def build_auditor_messages(
    base_claim: str,
    current_claim: str,
) -> List[Dict[str, str]]:
    
    system_prompt = f"""
    You are an expert evaluator applying FUSE-EVAL to measure how much a modified claim
    deviates from the original claim. You MUST respond with a strict JSON object only,
    no additional text.

    You will compare a BASE_CLAIM and a CURRENT_CLAIM using the following six dimensions:

    1. Sentiment Shift (SS):
    How does the sentiment of the claim compare to the original claim? Is the tone more positive,
    negative, or neutral compared to the original?

    2. New Information Introduced (NII):
    Does the claim introduce additional information not in the original claim, such as political
    conspiracy or speculation? Evaluate how much of the article is focused on these claim details.

    3. Certainty Shift (CS):
    How does the claim language change in terms of certainty? Does it use more ambiguous terms
    like “possibly” or “allegedly” compared to the original claim, or does it present the
    information with more certainty?

    4. Stylistic Shift (STS):
    How does the writing style compare to the original? Has the claim moved from neutral reporting
    to a more exaggerated or dramatic tone?

    5. Temporal Shift (TS):
    Does the claim shift focus from the specific event mentioned in the original claim to broader or
    unrelated timeframes, such as mentioning legal battles or long-term political issues?

    6. Perspective Deviation (PD):
    Does the claim introduce subjective opinions or perspectives that deviate from the objective
    reporting in the original claim? For instance, questioning the truth of the event or speculating
    on hidden motives.

    For each dimension, assign a deviation score from 0.0 to 10.0:
    - 0.0 means no noticeable deviation.
    - 10.0 means extremely strong deviation.
    
    You MUST output a valid JSON object with EXACTLY the following structure:
    {{
      "SS": {{"score": <float>, "explanation": "..."}}, 
      "NII": {{"score": <float>, "explanation": "..."}}, 
      "CS": {{"score": <float>, "explanation": "..."}}, 
      "STS": {{"score": <float>, "explanation": "..."}}, 
      "TS": {{"score": <float>, "explanation": "..."}}, 
      "PD": {{"score": <float>, "explanation": "..."}}
    }}
    """

    user_prompt = f"""
    BASE_CLAIM:
    {base_claim}

    CURRENT_CLAIM:
    {current_claim}

    Compare CURRENT_CLAIM to BASE_CLAIM and provide the FUSE-EVAL scores and explanations as JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def _extract_json_from_text(text: str) -> str:
   
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return text


def parse_auditor_response(raw_output: str) -> Dict[str, Any]:
    
    text = raw_output.strip()
    text = _extract_json_from_text(text)

    try:
        data = json.loads(text)
        dims = ["SS", "NII", "CS", "STS", "TS", "PD"]
        result: Dict[str, Any] = {}
        for d in dims:
            if d in data and isinstance(data[d], dict):
                score = data[d].get("score")
                explanation = data[d].get("explanation")
            else:
                score = None
                explanation = None
            result[d] = {
                "score": float(score) if isinstance(score, (int, float)) else score,
                "explanation": explanation,
            }
        return result
    except Exception:
        return {
            "parse_error": True,
            "raw_output": raw_output,
        }
