'''
cluster_network
'''

import json
import random
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

from openai import OpenAI

from cluster_processing_verdict import (
    build_character_messages,
    build_auditor_messages,
    parse_character_response,
    parse_auditor_response,
)

MODEL = "gpt-4o"
API_KEY = "xxxxxx"
BASE_URL = "https://example.org"


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


MAX_RETRIES = 3          
RETRY_BASE_DELAY = 2     

MAX_STEPS_PER_CLAIM = 10


def load_characters(characters_path: str) -> Dict[str, Dict[str, Any]]:
    with open(characters_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_cluster_network(
    network_path: str,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]]]:
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

    # cluster -> nodes
    cluster_to_nodes: Dict[str, List[str]] = {}

    for comm in data.get("communities", []):
        name = comm.get("name")
        if not name:
            continue
        nodes = comm.get("nodes", [])
        cluster_to_nodes.setdefault(name, [])
        for nid in nodes:
            if nid in adjacency:  
                cluster_to_nodes[name].append(nid)

    for n in data.get("nodes", []):
        nid = n.get("id")
        cname = n.get("community")
        if nid and cname:
            cluster_to_nodes.setdefault(cname, [])
            if nid not in cluster_to_nodes[cname]:
                cluster_to_nodes[cname].append(nid)

    return node_ids, adjacency, cluster_to_nodes


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


def _build_openai_client(base_url: str, api_key_env_default: Optional[str]) -> OpenAI:
    
    if base_url:
        client_ = OpenAI(base_url=base_url, api_key="DUMMY")  
    else:
        client_ = OpenAI(api_key="DUMMY")
    return client_


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
    cluster_name: str,
    cluster_node_ids: List[str],
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
    Simulate the propagation of a single base_claim over a maximum of max_steps steps within the "specified cluster" network. 
    Propagation rules (cluster version):
    - It is only propagated among the nodes corresponding to the cluster_node_ids;
    - The first step: randomly select a starting point among all the nodes in this cluster;
    - For each subsequent step:
    * From the neighbors of the current node, select "nodes that also belong to this cluster" randomly;
    * If there are no such eligible neighbors, randomly select a new starting point from the cluster nodes again (the number of steps still accumulates, not reset).
    """
    if not cluster_node_ids:
        raise ValueError(f"Cluster '{cluster_name}' has no nodes.")

    base_claim = base_claim_info["base_claim"]

    steps: List[Dict[str, Any]] = []

    dims = ["SS", "NII", "CS", "STS", "TS", "PD"]
    cumulative_sums = {d: 0.0 for d in dims}
    dim_avg_series = {d: [] for d in dims}
    fuse_cumulative_sum = 0.0
    fuse_avg_series: List[float] = []

    dim_step_scores = {d: [] for d in dims}
    fuse_step_scores: List[float] = []

    cluster_node_set = set(cluster_node_ids)

    # Initialization: Select a starting point randomly from this cluster
    current_node = random.choice(cluster_node_ids)
    current_claim = base_claim

    for step_idx in range(1, max_steps + 1):
        character_profile = characters.get(current_node, {})

        # 1. Character generates the rewritten claim
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
        # Parse and rewrite the claim and top_attributes (names) from the LLM output
        rewritten_claim, attribute_str = parse_character_response(char_raw_output)

        # 2. The auditor conducts the FUSE-EVAL assessment
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

        # Calculate the scores for each dimension of this step and the cumulative average
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

        # This step's FUSE-EVAL comprehensive score (simple average of six dimensions)
        if dims:
            step_fuse_score = sum(step_scores[d] for d in dims) / len(dims)
        else:
            step_fuse_score = 0.0
        fuse_cumulative_sum += step_fuse_score
        fuse_avg_series.append(fuse_cumulative_sum / step_idx)
        fuse_step_scores.append(step_fuse_score)

        # 3. Determine the next node to visit (only traverse edges within the same cluster)
        neighbors_all = adjacency.get(current_node, [])
        neighbors_in_cluster = [n for n in neighbors_all if n in cluster_node_set]

        if neighbors_in_cluster:
            next_node = random.choice(neighbors_in_cluster)
        else:
            # If there are no neighbors in the same cluster => Select a new random node within this cluster
            next_node = random.choice(cluster_node_ids)

        steps.append(
            {
                "step": step_idx,
                "producer_character": current_node,
                "next_character": next_node,
                "rewritten_claim": rewritten_claim,
                "attribute": attribute_str,
                "fuse_eval": fuse_eval_result,
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

    final_claim = current_claim

    result = {
        "cluster": cluster_name,
        "topic": base_claim_info.get("topic"),
        "title": base_claim_info.get("title"),
        "url": base_claim_info.get("url"),
        "article_index": base_claim_info.get("article_index"),
        "claim_index": base_claim_info.get("claim_index"),
        "base_claim": base_claim,
        "final_claim": final_claim,
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
        "FUSE-EVAL-max": _format_max_series(fuse_step_scores),
    }
    return result


# ========== Main program: Propagate each seed_claim in 3 clusters respectively. ==========

def main():
    random.seed(42)

    characters_path = "characters_group.json"
    network_path = "cluster_network.json"
    publisher_claims_path = "seed_claims.json"

    output_paths = {
        "Traditional Conservative": "Traditional_Conservative_results.json",
        "Technocratic Moderate": "Technocratic_Moderate_results.json",
        "Liberal Elite": "Liberal_Elite_results.json",
    }

    for path in output_paths.values():
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    cluster_order = [
        "Traditional Conservative",
        "Technocratic Moderate",
        "Liberal Elite",
    ]

    character_llm_base_url = ""
    character_llm_model = ""
    auditor_llm_base_url = ""
    auditor_llm_model = ""
    character_llm_api_key = None
    auditor_llm_api_key = None
    max_steps = 10

    characters = load_characters(characters_path)
    node_ids, adjacency, cluster_to_nodes = load_cluster_network(network_path)
    base_claims = load_base_claims(publisher_claims_path)

    cluster_results: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in cluster_order
    }

    for idx, claim_info in enumerate(base_claims):
        print(
            f"\n===== Base claim {idx + 1}/{len(base_claims)} "
            f"[topic={claim_info.get('topic')}, claim_index={claim_info.get('claim_index')}] ====="
        )

        for cluster_name in cluster_order:
            cluster_nodes = cluster_to_nodes.get(cluster_name, [])
            if not cluster_nodes:
                print(
                    f"[WARN] Cluster '{cluster_name}' has no nodes in network, skip for this claim.",
                    file=sys.stderr,
                )
                continue

            print(
                f"  -> Simulating in cluster: {cluster_name} "
                f"(steps={max_steps}, nodes={len(cluster_nodes)})"
            )

            result = simulate_claim_propagation(
                base_claim_info=claim_info,
                cluster_name=cluster_name,
                cluster_node_ids=cluster_nodes,
                adjacency=adjacency,
                characters=characters,
                character_llm_base_url=character_llm_base_url,
                character_llm_model=character_llm_model,
                auditor_llm_base_url=auditor_llm_base_url,
                auditor_llm_model=auditor_llm_model,
                character_llm_api_key=character_llm_api_key or None,
                auditor_llm_api_key=auditor_llm_api_key or None,
                max_steps=max_steps,
            )

            cluster_results[cluster_name].append(result)

            out_path = output_paths[cluster_name]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cluster_results[cluster_name], f, ensure_ascii=False, indent=2)

            print(
                f"Saved incremental results for cluster '{cluster_name}' "
                f"up to base claim {idx + 1}."
            )

    print("All done.")


if __name__ == "__main__":
    main()
