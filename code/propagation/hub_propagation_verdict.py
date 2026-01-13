'''
scale-free_network
'''

import json
import random
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

from openai import OpenAI

MODEL = "gpt-4o"
API_KEY = "xxxxxx"   
BASE_URL = "https://example.org"


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

from hub_processing_verdict import (
    build_character_messages,
    build_auditor_messages,
    parse_character_response,
    parse_auditor_response,
    build_hub_opinion_messages,
    parse_hub_opinion_response,
)

MAX_STEPS_PER_CLAIM = 10


def load_characters(characters_path: str) -> Dict[str, Dict[str, Any]]:
    with open(characters_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_network(network_path: str) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    
    with open(network_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_ids: List[str] = []
    hub_ids: List[str] = []

    for n in data.get("nodes", []):
        nid = n["id"]
        node_ids.append(nid)
        if n.get("hub"):
            hub_ids.append(nid)

    adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for edge in data.get("edges", []):
        s = edge["source"]
        t = edge["target"]
        if s in adjacency and t in adjacency:
            adjacency[s].append(t)
            adjacency[t].append(s)

    return node_ids, adjacency, hub_ids


def load_base_claims(publisher_claims_path: str) -> List[Dict[str, Any]]:
    with open(publisher_claims_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results: List[Dict[str, Any]] = []

    for topic, articles in data.items():
        if not isinstance(articles, list):
            continue
        for article_idx, article in enumerate(articles):
            title = article.get("title")
            url = article.get("url")
            claims = article.get("claims", [])
            if not claims:
                continue
            for claim_idx, claim in enumerate(claims):
                base_claim = claim.get("generated_claim")
                if not base_claim:
                    continue
                results.append(
                    {
                        "topic": topic,
                        "article_index": article_idx,
                        "claim_index": claim_idx,
                        "title": title,
                        "url": url,
                        "base_claim": base_claim,
                        "original_claim_meta": claim,
                    }
                )

    return results


def choose_random_start_node(node_ids: List[str]) -> str:
    return random.choice(node_ids)


def choose_next_node(
    current_node: str, adjacency: Dict[str, List[str]], node_ids: List[str]
) -> str:
    
    neighbors = adjacency.get(current_node, [])
    if neighbors:
        return random.choice(neighbors)
    return choose_random_start_node(node_ids)


def choose_random_hub(hub_ids: List[str]) -> Optional[str]:
    if not hub_ids:
        return None
    return random.choice(hub_ids)


def character_llm_call(
    messages: List[Dict[str, str]],
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
) -> str:
    used_model = model or MODEL

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=used_model,
                messages=messages,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            print(
                f"[Character LLM ERROR] attempt {attempt}/{MAX_RETRIES} failed: {e}",
                file=sys.stderr,
            )
            if attempt == MAX_RETRIES:
                return ""
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[Character LLM RETRY] sleep {delay} seconds...", file=sys.stderr)
            time.sleep(delay)


def auditor_llm_call(
    messages: List[Dict[str, str]],
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
) -> str:
    used_model = model or MODEL

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=used_model,
                messages=messages,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            print(
                f"[Auditor LLM ERROR] attempt {attempt}/{MAX_RETRIES} failed: {e}",
                file=sys.stderr,
            )
            if attempt == MAX_RETRIES:
                return ""
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[Auditor LLM RETRY] sleep {delay} seconds...", file=sys.stderr)
            time.sleep(delay)


def simulate_claim_propagation(
    base_claim_info: Dict[str, Any],
    node_ids: List[str],
    adjacency: Dict[str, List[str]],
    hub_ids: List[str],
    characters: Dict[str, Dict[str, Any]],
    character_llm_base_url: str,
    character_llm_model: str,
    auditor_llm_base_url: str,
    auditor_llm_model: str,
    character_llm_api_key: Optional[str] = None,
    auditor_llm_api_key: Optional[str] = None,
    max_steps: int = MAX_STEPS_PER_CLAIM,
) -> Dict[str, Any]:
    """
    For the propagation of a single base_claim over a scale-free network for a maximum of max_steps:
    - At each step, randomly select a hub, and generate 1-2 viewpoints (hub_opinion) for the current_claim based on the hub
    - The current node is influenced by the hub_opinion when rewriting the claim
    - Only randomly select the next node from the neighbors; if there are no neighbors, randomly select a new node from the entire network to continue
    """
    base_claim = base_claim_info["base_claim"]

    steps: List[Dict[str, Any]] = []

    dims = ["SS", "NII", "CS", "STS", "TS", "PD"]
    dim_step_scores = {d: [] for d in dims}
    fuse_step_scores: List[float] = []

    # hub perspective cache: Prevent repeated calls for the combination (hub_id, current_claim)
    hub_opinion_cache: Dict[Tuple[str, str], str] = {}

    current_node = choose_random_start_node(node_ids)
    current_claim = base_claim

    for step_idx in range(1, max_steps + 1):
        character_profile = characters.get(current_node, {})

        # 1. Select a hub and generate an opinion on the "current_claim"
        influencing_hub_id = choose_random_hub(hub_ids)
        hub_opinion = None
        if influencing_hub_id is not None:
            cache_key = (influencing_hub_id, current_claim)
            if cache_key in hub_opinion_cache:
                hub_opinion = hub_opinion_cache[cache_key]
            else:
                hub_profile = characters.get(influencing_hub_id, {})
                hub_messages = build_hub_opinion_messages(
                    hub_id=influencing_hub_id,
                    hub_profile=hub_profile,
                    current_claim=current_claim,
                )
                hub_raw_output = character_llm_call(
                    messages=hub_messages,
                    base_url=character_llm_base_url,
                    model=character_llm_model,
                    api_key=character_llm_api_key,
                )
                hub_opinion = parse_hub_opinion_response(hub_raw_output)
                hub_opinion_cache[cache_key] = hub_opinion

        # 2. The current node has rewritten the claim, and it is influenced by hub_opinion.
        char_messages = build_character_messages(
            character_id=current_node,
            character_profile=character_profile,
            base_claim=base_claim,
            current_claim=current_claim,
            influencing_hub_id=influencing_hub_id,
            influencing_hub_opinion=hub_opinion,
        )

        char_raw_output = character_llm_call(
            messages=char_messages,
            base_url=character_llm_base_url,
            model=character_llm_model,
            api_key=character_llm_api_key,
        )
        rewritten_claim, attribute_str = parse_character_response(char_raw_output)

        # 3. FUSE-EVAL Evaluation
        auditor_messages = build_auditor_messages(
            base_claim=base_claim,
            current_claim=rewritten_claim,
        )

        auditor_raw_output = auditor_llm_call(
            messages=auditor_messages,
            base_url=auditor_llm_base_url,
            model=auditor_llm_model,
            api_key=auditor_llm_api_key,
        )
        fuse_eval_result = parse_auditor_response(auditor_raw_output)

        step_scores: Dict[str, float] = {}
        for d in dims:
            raw_dim = fuse_eval_result.get(d) if isinstance(fuse_eval_result, dict) else None
            if isinstance(raw_dim, dict):
                score = raw_dim.get("score")
            else:
                score = None
            s = float(score) if isinstance(score, (int, float)) else 0.0
            step_scores[d] = s
            dim_step_scores[d].append(s)

        if dims:
            step_fuse_score = sum(step_scores[d] for d in dims) / len(dims)
        else:
            step_fuse_score = 0.0
        fuse_step_scores.append(step_fuse_score)

        # 4. Decide on the next node
        next_node = choose_next_node(current_node, adjacency, node_ids)

        steps.append(
            {
                "step": step_idx,
                "producer_character": current_node,
                "next_character": next_node,
                "influencing_hub": influencing_hub_id,
                "hub_opinion": hub_opinion,
                "rewritten_claim": rewritten_claim,
                "attribute": attribute_str,
                "fuse_eval": fuse_eval_result,
            }
        )

        current_node = next_node
        current_claim = rewritten_claim

    def _format_series(series: List[float]) -> str:
        return "; ".join(f"{v:.1f}" for v in series)

    def _format_max_series(series: List[float]) -> str:
        if not series:
            return ""
        max_val = max(series)
        max_step = series.index(max_val) + 1
        return f"{max_val:.1f}; {max_step}"

    result = {
        "topic": base_claim_info.get("topic"),
        "title": base_claim_info.get("title"),
        "url": base_claim_info.get("url"),
        "article_index": base_claim_info.get("article_index"),
        "claim_index": base_claim_info.get("claim_index"),
        "base_claim": base_claim,
        "original_claim_meta": base_claim_info.get("original_claim_meta"),
        "propagation_steps": steps,
        "SS": _format_series(dim_step_scores["SS"]),
        "NII": _format_series(dim_step_scores["NII"]),
        "CS": _format_series(dim_step_scores["CS"]),
        "STS": _format_series(dim_step_scores["STS"]),
        "TS": _format_series(dim_step_scores["TS"]),
        "PD": _format_series(dim_step_scores["PD"]),
        "FUSE-EVAL": _format_series(fuse_step_scores),
        "SS-max": _format_max_series(dim_step_scores["SS"]),
        "NII-max": _format_max_series(dim_step_scores["NII"]),
        "CS-max": _format_max_series(dim_step_scores["CS"]),
        "STS-max": _format_max_series(dim_step_scores["STS"]),
        "TS-max": _format_max_series(dim_step_scores["TS"]),
        "PD-max": _format_max_series(dim_step_scores["PD"]),
        "FUSE-EVAL-max": _format_max_series(fuse_step_scores),
    }
    return result


def main():
    random.seed(42)

    characters_path = "characters_group.json"
    network_path = "scale_free_network.json"
    publisher_claims_path = "seed_claims.json"
    output_path = "hub_propagation_verdict_results.json"

    character_llm_base_url = ""
    character_llm_model = ""
    auditor_llm_base_url = ""
    auditor_llm_model = ""
    character_llm_api_key = None
    auditor_llm_api_key = None
    max_steps = 10

    characters = load_characters(characters_path)
    node_ids, adjacency, hub_ids = load_network(network_path)
    base_claims = load_base_claims(publisher_claims_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    for idx, claim_info in enumerate(base_claims):
        print(
            f"Simulating propagation for claim {idx + 1}/{len(base_claims)} "
            f"[topic={claim_info.get('topic')}, claim_index={claim_info.get('claim_index')}]"
        )
        result = simulate_claim_propagation(
            base_claim_info=claim_info,
            node_ids=node_ids,
            adjacency=adjacency,
            hub_ids=hub_ids,
            characters=characters,
            character_llm_base_url=character_llm_base_url,
            character_llm_model=character_llm_model,
            auditor_llm_base_url=auditor_llm_base_url,
            auditor_llm_model=auditor_llm_model,
            character_llm_api_key=character_llm_api_key or None,
            auditor_llm_api_key=auditor_llm_api_key or None,
            max_steps=max_steps,
        )

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

        existing.append(result)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
