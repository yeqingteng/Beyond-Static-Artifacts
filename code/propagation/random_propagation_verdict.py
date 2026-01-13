'''
random_network
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

from random_processing_verdict import (
    build_character_messages,
    build_auditor_messages,
    parse_character_response,
    parse_auditor_response,
)

MAX_STEPS_PER_CLAIM = 10

def load_characters(characters_path: str) -> Dict[str, Dict[str, Any]]:
    
    with open(characters_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_network(network_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    
    with open(network_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_ids = [n["id"] for n in data.get("nodes", [])]
    adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for edge in data.get("edges", []):
        s = edge["source"]
        t = edge["target"]
        if s in adjacency and t in adjacency:
            adjacency[s].append(t)
            adjacency[t].append(s)  

    return node_ids, adjacency


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
    """
    Randomly select the next node from the neighbors of the current node;
    If there are no neighbors (no way to go), randomly reselect the starting point from the entire network.
    """
    neighbors = adjacency.get(current_node, [])
    if neighbors:
        return random.choice(neighbors)
    return choose_random_start_node(node_ids)


def _build_openai_client(base_url: str, api_key_env_default: Optional[str]) -> OpenAI:
    
    if base_url:
        client = OpenAI(base_url=base_url, api_key="DUMMY")  
    else:
        client = OpenAI(api_key="DUMMY")
    return client


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

def get_top_attributes(character_profile: Dict[str, Any]) -> str:
    """
    Select the top three attributes (names) that influence the rewriting of the character from the attribute list; 
    use ";" connection
    For example: "religious;" ideology; area"
    """
    ordered_keys = [
        "religious","ideology","area","age","gender","employment",
        "income","education","job","marital","big_five"
    ]

    attrs = []
    for k in ordered_keys:
        if k in character_profile and character_profile[k] not in [None, "", []]:
            attrs.append(k)

    return "; ".join(attrs[:3])

def simulate_claim_propagation(
    base_claim_info: Dict[str, Any],
    node_ids: List[str],
    adjacency: Dict[str, List[str]],
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
    Simulate the propagation of a single base_claim within the network for a maximum of max_steps (default 10 steps)
    At each step:
    - The current node modifies the claim based on its own attributes
    - The Auditor uses FUSE-EVAL to evaluate the deviation of this modification from the base_claim
    - Randomly select a neighbor as the next Producer; if there are no neighbors, randomly choose a new starting point (but still accumulate the number of steps)
    """
    base_claim = base_claim_info["base_claim"]

    steps: List[Dict[str, Any]] = []

    dims = ["SS", "NII", "CS", "STS", "TS", "PD"]
    cumulative_sums = {d: 0.0 for d in dims}
    dim_avg_series = {d: [] for d in dims}
    fuse_cumulative_sum = 0.0
    fuse_avg_series: List[float] = []

    dim_step_scores = {d: [] for d in dims}
    fuse_step_scores: List[float] = []

    current_node = choose_random_start_node(node_ids)
    current_claim = base_claim

    for step_idx in range(1, max_steps + 1):
        character_profile = characters.get(current_node, {})

        char_messages = build_character_messages(
            character_id=current_node,
            character_profile=character_profile,
            base_claim=base_claim,
            current_claim=current_claim,
        )

        char_raw_output = character_llm_call(
            messages=char_messages,
            base_url=character_llm_base_url,
            model=character_llm_model,
            api_key=character_llm_api_key,
        )
        rewritten_claim, attribute_str = parse_character_response(char_raw_output)

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
            cumulative_sums[d] += s
            dim_avg_series[d].append(cumulative_sums[d] / step_idx)
            dim_step_scores[d].append(s)

        if dims:
            step_fuse_score = sum(step_scores[d] for d in dims) / len(dims)
        else:
            step_fuse_score = 0.0
        fuse_cumulative_sum += step_fuse_score
        fuse_avg_series.append(fuse_cumulative_sum / step_idx)
        fuse_step_scores.append(step_fuse_score)


        next_node = choose_next_node(current_node, adjacency, node_ids)

        steps.append(
            {
                "step": step_idx,
                "producer_character": current_node,
                "next_character": next_node,
                "rewritten_claim": rewritten_claim,
                "attribute": attribute_str,
                "fuse_eval": fuse_eval_result
            }
        )

        
        current_node = next_node
        current_claim = rewritten_claim

    
    def _format_avg_series(series: List[float]) -> str:
        
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
        "SS": _format_avg_series(dim_step_scores["SS"]),
        "NII": _format_avg_series(dim_step_scores["NII"]),
        "CS": _format_avg_series(dim_step_scores["CS"]),
        "STS": _format_avg_series(dim_step_scores["STS"]),
        "TS": _format_avg_series(dim_step_scores["TS"]),
        "PD": _format_avg_series(dim_step_scores["PD"]),
        "FUSE-EVAL": _format_avg_series(fuse_step_scores),
        "SS-max": _format_max_series(dim_step_scores["SS"]),
        "NII-max": _format_max_series(dim_step_scores["NII"]),
        "CS-max": _format_max_series(dim_step_scores["CS"]),
        "STS-max": _format_max_series(dim_step_scores["STS"]),
        "TS-max": _format_max_series(dim_step_scores["TS"]),
        "PD-max": _format_max_series(dim_step_scores["PD"]),
        "FUSE-EVAL-max": _format_max_series(fuse_step_scores)
    }
    return result

def main():
    random.seed(42)

    characters_path = "characters_group.json"
    network_path = "random_network.json"
    publisher_claims_path = "seed_claims.json"
    output_path = "random_propagation_verdict_results.json"

    character_llm_base_url = ""
    character_llm_model = ""
    auditor_llm_base_url = ""
    auditor_llm_model = ""
    character_llm_api_key = None
    auditor_llm_api_key = None
    max_steps = 10

    characters = load_characters(characters_path)
    node_ids, adjacency = load_network(network_path)
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
            characters=characters,
            character_llm_base_url=character_llm_base_url,
            character_llm_model=character_llm_model,
            auditor_llm_base_url=auditor_llm_base_url,
            auditor_llm_model=auditor_llm_model,
            character_llm_api_key=character_llm_api_key or None,
            auditor_llm_api_key=auditor_llm_api_key or None,
            max_steps=max_steps
        )
        #all_results.append(result)

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

    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
