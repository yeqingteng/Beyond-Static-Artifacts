# -*- coding: utf-8 -*-
"""
This document is responsible for the overall dissemination and review process:
1) Reading character attributes; Network structure (undirected graph); seed-claims;
2) Propagate for each seed-claim and each non-any customized dimension;
3) Run a separate propagation chain for each dimension, with a maximum of 10 steps;
4) At each step: Node processing + review;
5) If the review is passed, stop immediately; otherwise, mark this dimension as failure;
"""
import json
import random
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

from node_processing import (
    process_character_node,
    audit_dimension,
    DIMENSION_DESCRIPTIONS,
    call_llm  
)


CHARACTERS_PATH = "characters_group.json"
NETWORK_PATH = "random_network.json"
PUBLISHER_CLAIMS_PATH = "seed_claims.json"
OUTPUT_PATH = "intervened_propagation_results.json"
CLIENT_CONFIG_PATH = None                          


DEFAULT_CLIENT_CONFIG: Dict[str, str] = {
    "linguistic_style": "Rational–Official",

    "message_type": "Fact",

    "granularity": "Macro/High-Level",

    "causal_structure": "Direct Causation",

    "stance_polarity": "Support"
}

DIMENSION_ORDER = [
    "linguistic_style",
    "message_type",
    "granularity",
    "causal_structure",
    "stance_polarity"
]


def select_best_character_by_attributes(
    candidate_ids: List[str],
    characters: Dict[str, Dict[str, Any]],
    dimension: str,
    target_value: str,
    auditor2_feedback: Dict[str, Any] = None,
    auditor3_feedback: Dict[str, Any] = None,
) -> str:
    if not candidate_ids:
        return None

    if len(candidate_ids) == 1:
        return candidate_ids[0]

    candidates_info: Dict[str, Any] = {}
    for cid in candidate_ids:
        candidates_info[cid] = characters.get(cid, {"id": cid})

    dim_cfg = DIMENSION_DESCRIPTIONS.get(dimension, {})
    dim_label = dim_cfg.get("label", dimension)
    human_value = dim_cfg.get("values", {}).get(target_value, target_value)

    feedback2 = auditor2_feedback or {}
    feedback3 = auditor3_feedback or {}

    prompt = f"""
You are selecting the next node in an information propagation network.

[Target customization dimension]
- Internal name: {dimension}
- Label: {dim_label}
- Target value (internal): {target_value}
- Target value (human-readable): {human_value}

[Auditor feedback about why the current claim still does NOT fully satisfy the requirement]
- auditor2_pass: {bool(feedback2.get("pass", False))}
- auditor2_reason: {feedback2.get("reason", "")}
- auditor3_pass: {bool(feedback3.get("pass", False))}
- auditor3_reason: {feedback3.get("reason", "")}

[Candidate characters]
The following JSON maps candidate node IDs to their attributes:
{json.dumps(candidates_info, ensure_ascii=False, indent=2)}

Your task:
- Based on the above dimension requirement and the candidate attributes,
  choose ONE candidate who is most suitable to further rewrite or handle
  the current claim so that it moves closer to the target value on THIS dimension.
- Think of "suitable" as: their attributes make them likely to produce text that better matches this dimension.

Output requirements:
- ONLY output a JSON object with exactly one field:
  - "best_character_id": string, one of the candidate IDs.

Example:
{{"best_character_id": "23"}}
"""

    raw = call_llm(prompt)
    try:
        data = json.loads(raw)
        cid = data.get("best_character_id")
        if cid in candidate_ids:
            return cid
    except Exception:
        pass

    return random.choice(candidate_ids)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_undirected_adjacency(network_data: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:
    nodes = [node["id"] for node in network_data.get("nodes", [])]
    adj: Dict[str, List[str]] = {nid: [] for nid in nodes}

    for edge in network_data.get("edges", []):
        src = edge.get("source")
        tgt = edge.get("target")

        if src is not None and tgt is not None:
            
            if src not in adj:
                adj[src] = []
            if tgt not in adj[src]:
                adj[src].append(tgt)

            if tgt not in adj:
                adj[tgt] = []
            if src not in adj[tgt]:
                adj[tgt].append(src)

    return nodes, adj

def propagate_single_dimension(
    base_claim: str,
    dimension: str,
    target_value: str,
    nodes: List[str],
    adjacency: Dict[str, List[str]],
    characters: Dict[str, Dict[str, Any]],
    article_content: str,
    max_steps: int = 10,
    max_rounds: int = 3,
) -> Optional[Dict[str, Any]]:
    
    if target_value == "any":
        return None

    if not nodes:
        return {
            "dimension": dimension,
            "target_value": target_value,
            "status": "failure",
            "final_claim": base_claim,
            "steps": [],
            "auditor_rounds": [],
        }

    all_steps_log: List[Dict[str, Any]] = []
    auditor_round_logs: List[Dict[str, Any]] = []

    final_claim = base_claim
    status = "failure"
    majority_satisfied = False

    next_start_node: str = None  

    last_audit_feedback2: Dict[str, Any] = None  
    last_audit_feedback3: Dict[str, Any] = None  

    for round_idx in range(1, max_rounds + 1):
        current_claim = final_claim if round_idx > 1 else base_claim
        current_node = next_start_node
        steps_used = 0
        satisfied_by_auditor1 = False

        directed_mode = round_idx > 1

        round_step_indices: List[int] = []

        while steps_used < max_steps:
            if current_node is None or len(adjacency.get(current_node, [])) == 0:
                if directed_mode:
                    if steps_used == 0 and next_start_node is not None:
                        current_node = next_start_node
                    else:
                        current_node = select_best_character_by_attributes(
                            candidate_ids=nodes,
                            characters=characters,
                            dimension=dimension,
                            target_value=target_value,
                            auditor2_feedback=last_audit_feedback2,
                            auditor3_feedback=last_audit_feedback3,
                        )
                else:
                    current_node = random.choice(nodes)

            char_attrs = characters.get(current_node, {"name": current_node})
            input_claim = current_claim

            node_result = process_character_node(
                character_id=current_node,
                attributes=char_attrs,
                claim_text=current_claim,
                dimension=dimension,
                target_value=target_value,
                base_claim=base_claim,
                article_content=article_content,
            )
            updated_claim = node_result.get("updated_claim", current_claim)
            role = node_result.get("role", "bystander")
            #character_reason = node_result.get("character_reason", "")
            influencing_attributes = node_result.get("influencing_attributes", [])
            influencing_reason = node_result.get("influencing_reason", "")

            steps_used += 1
            current_claim = updated_claim

            # Auditor 1 conducts the audit
            audit_result1 = audit_dimension(
                claim_text=current_claim,
                dimension=dimension,
                target_value=target_value,
                auditor_id="auditor1",
            )
            audit_pass1 = bool(audit_result1.get("pass", False))
            audit_reason1 = audit_result1.get("reason", "")

            step_record = {
                "round": round_idx,
                "step": steps_used,
                "character_id": current_node,
                "role": role,
                "influencing_attributes": influencing_attributes,
                "influencing_reason": influencing_reason,
                "input_claim": input_claim,
                "output_claim": current_claim,
                "audit_pass": audit_pass1,
                "audit_reason": audit_reason1,
            }
            all_steps_log.append(step_record)
            round_step_indices.append(len(all_steps_log) - 1)

            if audit_pass1:
                satisfied_by_auditor1 = True
                break

            neighbors = adjacency.get(current_node, [])
            if neighbors:
                if directed_mode:
                    current_node = select_best_character_by_attributes(
                        candidate_ids=neighbors,
                        characters=characters,
                        dimension=dimension,
                        target_value=target_value,
                        auditor2_feedback=last_audit_feedback2,
                        auditor3_feedback=last_audit_feedback3,
                    )
                else:
                    current_node = random.choice(neighbors)
            else:
                current_node = None

        final_claim = current_claim
        # ========== Majority Vote of Three Auditors ==========
        # Regardless of whether auditor1 passed at any step in this round, a joint assessment by all three auditors will be conducted:
        # The result of auditor1 is taken as the review result of the last step of this round;
        # Auditors 2 and 3 conduct additional reviews on the final_claim;
        # If ≥ 2 auditors determine "not passed", then proceed to the next round of targeted dissemination;
        # Otherwise, it is considered that this dimension has been satisfied and the process can be directly concluded successfully.
        if round_step_indices:
            last_step_idx = round_step_indices[-1]
            last_step = all_steps_log[last_step_idx]
            pass1 = bool(last_step.get("audit_pass", False))
            auditor1_reason = last_step.get("audit_reason", "")
        else:
            pass1 = False
            auditor1_reason = ""

        audit_result2 = audit_dimension(
            claim_text=final_claim,
            dimension=dimension,
            target_value=target_value,
            auditor_id="auditor2",
        )
        audit_result3 = audit_dimension(
            claim_text=final_claim,
            dimension=dimension,
            target_value=target_value,
            auditor_id="auditor3",
        )

        pass2 = bool(audit_result2.get("pass", False))
        pass3 = bool(audit_result3.get("pass", False))

        auditor_round_logs.append(
            {
                "round": round_idx,
                "auditor1": {"pass": pass1, "reason": auditor1_reason},
                "auditor2": {
                    "pass": pass2,
                    "reason": audit_result2.get("reason", ""),
                },
                "auditor3": {
                    "pass": pass3,
                    "reason": audit_result3.get("reason", ""),
                },
            }
        )

        # The number of auditors whose assessment was not passed:
        # If ≥ 2 auditors determined it as not passing → Proceed to the subsequent rounds (if there are still remaining rounds);
        # Otherwise, consider it a success and terminate all rounds immediately.
        num_pass = int(pass1) + int(pass2) + int(pass3)
        num_fail = 3 - num_pass

        if num_fail < 2:
            # Most people think it's approved. This dimension is successful. End all rounds.
            majority_satisfied = True
            status = "success"
            break

        last_audit_feedback2 = audit_result2
        last_audit_feedback3 = audit_result3

        next_start_node = select_best_character_by_attributes(
            candidate_ids=nodes,
            characters=characters,
            dimension=dimension,
            target_value=target_value,
            auditor2_feedback=audit_result2,
            auditor3_feedback=audit_result3,
        )

    # The round is over. If a majority approval has not been achieved, the status will remain as "failure".
    return {
        "dimension": dimension,
        "target_value": target_value,
        "status": status,
        "final_claim": final_claim,
        "steps": all_steps_log,
        "auditor_rounds": auditor_round_logs,
    }

def process_all_claims(
    characters_path: str,
    network_path: str,
    publisher_claims_path: str,
    client_config: Dict[str, str],
    output_path: str
) -> None:
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

    
    characters = load_json(characters_path)             
    network_data = load_json(network_path)              
    publisher_data = load_json(publisher_claims_path)   

    nodes, adjacency = build_undirected_adjacency(network_data)

    total_claims_to_process = 0
    for _topic, articles in publisher_data.items():
        for article in articles:
            claims = article.get("claims", [])
            if not isinstance(claims, list):
                continue

            for claim_obj in claims:
                base_claim = claim_obj.get("generated_claim")
                if not base_claim:
                    continue

                has_active_dim = any(
                    client_config.get(dim, "any") != "any"
                    for dim in DIMENSION_ORDER
                )
                if has_active_dim:
                    total_claims_to_process += 1

    if total_claims_to_process == 0:
        print("[WARN] No generated_claim to process under current client_config.")
    else:
        print(f"[INFO] Total generated_claim to process: {total_claims_to_process}")

    results: Dict[str, List[Dict[str, Any]]] = {}
    pbar = tqdm(
        total=total_claims_to_process,
        desc="Processing generated_claim",
        unit="claim"
    ) if total_claims_to_process > 0 else None

    for topic, articles in publisher_data.items():
        topic_results: List[Dict[str, Any]] = []

        for article in articles:
            title = article.get("title", "")
            claims = article.get("claims", [])

            if not isinstance(claims, list):
                continue

            for claim_obj in claims:
                base_claim = claim_obj.get("generated_claim")
                if not base_claim:
                    continue

                has_active_dim = any(
                    client_config.get(dim, "any") != "any"
                    for dim in DIMENSION_ORDER
                )
                if not has_active_dim:
                    continue

                # Extract the content (the entire article content) corresponding to this claim from the current article
                article_content = article.get("content", "")

                propagations: List[Dict[str, Any]] = []

                # Propagate separately for each non-any dimension
                for dim in DIMENSION_ORDER:
                    target_val = client_config.get(dim, "any")

                    # "any" indicates that no customization is required for this dimension, and it will be skipped.
                    if target_val == "any":
                        continue

                    dim_result = propagate_single_dimension(
                        base_claim=base_claim,
                        dimension=dim,
                        target_value=target_val,
                        nodes=nodes,
                        adjacency=adjacency,
                        characters=characters,
                        article_content=article_content,
                        max_steps=10,
                        max_rounds=3
                    )
                    if dim_result is None:
                        continue
                    
                    propagations.append(dim_result)

                if propagations:
                    topic_results.append({
                        "article_title": title,
                        "base_claim": base_claim,
                        "propagations": propagations
                    })

                    results[topic] = topic_results
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

                if pbar is not None:
                    pbar.update(1)

        results[topic] = topic_results

    if pbar is not None:
        pbar.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Results written to: {output_path}")


def load_client_config() -> Dict[str, str]:
    
    config = DEFAULT_CLIENT_CONFIG.copy()
    if CLIENT_CONFIG_PATH is None:
        return config

    try:
        user_cfg = load_json(CLIENT_CONFIG_PATH)
        if isinstance(user_cfg, dict):
            for k, v in user_cfg.items():
                if k in config:
                    config[k] = v
    except FileNotFoundError:
        print(f"[WARN] Client config file not found: {CLIENT_CONFIG_PATH}. Using default config.")
    except Exception as e:
        print(f"[WARN] Error loading client config: {e}. Using default config.")

    return config

def main():
    
    random.seed(42)  

    client_config = load_client_config()
    print("[INFO] Using client_config:")
    print(json.dumps(client_config, ensure_ascii=False, indent=2))

    process_all_claims(
        characters_path=CHARACTERS_PATH,
        network_path=NETWORK_PATH,
        publisher_claims_path=PUBLISHER_CLAIMS_PATH,
        client_config=client_config,
        output_path=OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
