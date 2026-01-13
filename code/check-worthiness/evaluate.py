import json
from typing import Dict, Any, List, Tuple, Optional, Set

INPUT_FILE = r"intervened_propagation_evolved_results.json"

CHARACTERS_FILE = r"characters_group.json"

OUTPUT_FILE = r"intervened_propagation_evolved_worthiness.jsonl"

# ---------(1) Example of role attribute filtering conditions---------
# For instance: Only retain the roles whose "ideology" attribute is "liberal"
# If you don't want to filter a certain attribute, you can delete the key or set it to an empty dictionary {}
ATTRIBUTE_FILTERS: Dict[str, Any] = {
    # "ideology": "liberal",
}

# ---------(2) Dimension Filtering: Dimensions to be retained ---------
# For example: Only retain Q1-Q4
KEPT_Q_DIMENSIONS: List[str] = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

# Maximum Depth of Spread
MAX_PROPAGATION_DEPTH = 3

# The weights of the two internal indicators of propagation capacity (default values are both 0.5)
DEPTH_SCORE_WEIGHT = 0.5
SCALE_SCORE_WEIGHT = 0.5

# The weights of "propagation capability score" and "content value score" (both set to 0.5 by default)
PROPAGATION_VS_CONTENT_WEIGHT = 0.5

Q_DIMENSION_WEIGHTS = {q: 1 for q in ["Q1","Q2","Q3","Q4","Q5","Q6"]}

# If you want to set custom weights for each Q dimension, please do it here.
# If you only use Q1 and Q2 and want each to be 0.5:
# Q_DIMENSION_WEIGHTS = {"Q1": 0.5, "Q2": 0.5}

def load_characters(characters_file: str) -> Dict[str, Dict[str, Any]]:
    
    with open(characters_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_characters_by_attributes(
    characters: Dict[str, Dict[str, Any]],
    attribute_filters: Dict[str, Any]
) -> Set[str]:
    """
    Filter roles based on attributes:
    - attribute_filters are in the form of {"ideology": "liberal", "gender": "female"}
    - Return the set of role IDs that meet all the conditions
    """
    if not attribute_filters:
        # No restrictions - return all characters
        return set(characters.keys())

    matched = set()
    for char_id, attrs in characters.items():
        ok = True
        for key, value in attribute_filters.items():
            if key not in attrs or attrs[key] != value:
                ok = False
                break
        if ok:
            matched.add(char_id)
    return matched


def load_input_records(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data



def extract_nodes_from_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_nodes = record.get("propagation_log", [])

    ALL_Q_KEYS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

    nodes: List[Dict[str, Any]] = []
    for step in raw_nodes:
        char_id = step.get("character_id") or step.get("character")

        round_idx = step.get("round")

        scores = {}
        for q in ALL_Q_KEYS:
            if q in step:
                scores[q] = step[q]
        
        if "scores" in step and isinstance(step["scores"], dict):
            for q in ALL_Q_KEYS:
                if q in step["scores"] and q not in scores:
                    scores[q] = step["scores"][q]

        nodes.append({
            "character_id": char_id,
            "round": round_idx,
            "scores": scores,
        })

    return nodes


def filter_nodes_by_characters(
    nodes: List[Dict[str, Any]],
    allowed_char_ids: Set[str]
) -> List[Dict[str, Any]]:
    """
    In the transmission path, only the nodes whose role IDs belong to the "allowed_char_ids" are retained.
    """
    return [
        node for node in nodes
        if node.get("character_id") in allowed_char_ids
    ]


def compute_propagation_metrics(
    filtered_nodes: List[Dict[str, Any]],
    all_matched_char_ids: Set[str],
    max_depth: int,
    depth_weight: float = 0.5,
    scale_weight: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate the three related quantities related to the transmission ability:
    - Quantified transmission depth (0-1, one decimal place)
    - Quantified transmission scale (0-1, one decimal place)
    - Final transmission ability score (0-1, one decimal place), which is the weighted average of the above two indicators 

    The quantified propagation depth = Propagation depth in the filtered group / Maximum propagation depth
    - Here, "propagation depth in the filtered group" is defined as: The maximum value of rounds in the filtered_nodes
    - If numbered from 0, 1, 2, etc., then here, the maximum round is directly divided by 3 

    The quantified spread scale = Spread scale in the filtered group / Maximum spread scale
    - Numerator: The number of characters currently appearing after filtering (the number of unique characters)
    - Denominator: The total number of characters that meet the ATTRIBUTE_FILTERS conditions in characters_group.json
    """
    # ---- depth ----
    depth_score = 0.0
    rounds = [
        node["round"] for node in filtered_nodes
        if isinstance(node.get("round"), (int, float))
    ]
    if rounds and max_depth > 0:
        raw_depth = max(rounds)
        if raw_depth < 0:
            raw_depth = 0
        depth_score = round(min(raw_depth / max_depth, 1.0), 1)
    else:
        depth_score = 0.0

    # ---- size ----
    scale_score = 0.0
    # Filtered Set of Characters
    current_chars = {
        node.get("character_id") for node in filtered_nodes
        if node.get("character_id") is not None
    }
    # Maximum propagation scale = The total number of characters that meet the attribute filtering conditions among all the characters
    max_scale = len(all_matched_char_ids)
    if max_scale > 0:
        raw_scale = len(current_chars)
        scale_score = round(min(raw_scale / max_scale, 1.0), 1)
    else:
        scale_score = 0.0

    # ---- Final Propagation Ability Score (weighted average of two indicators) ----
    propagation_ability_score = round(
        depth_weight * depth_score + scale_weight * scale_score,
        1
    )

    return depth_score, scale_score, propagation_ability_score


def compute_content_value_score(
    filtered_nodes: List[Dict[str, Any]],
    kept_q_dims: List[str],
    q_weights: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, float], float]:
    """
    Calculate the content value score:
    - Step 1: In the filtered group, calculate the average value of each dimension in kept_q_dims (rounded to 1 decimal place)
    - Step 2: Perform a weighted average of these dimension average values to obtain the overall "content value score" (0-1, with 1 decimal place) 

    Return:
    - q_avg_dict: {Q dimension: The average value of this dimension in the filtered group}
    - content_value_score: The total content value score
    """
    q_avg_dict: Dict[str, float] = {}

    # First, calculate the average of each Q dimension within the filtered_nodes.
    for q in kept_q_dims:
        values: List[float] = []
        for node in filtered_nodes:
            scores = node.get("scores", {})
            v = scores.get(q, None)
            if isinstance(v, (int, float)):
                values.append(float(v))
        if values:
            avg = sum(values) / len(values)
            avg = round(avg, 1)  
        else:
            avg = 0.0
        q_avg_dict[q] = avg

    # Calculate weighted average
    if not kept_q_dims:
        return q_avg_dict, 0.0

    # If no weights are provided, then weights will be evenly distributed across all Q dimensions (with the sum equal to 1)
    if not q_weights:
        weight_each = 1.0 / len(kept_q_dims)
        weights = {q: weight_each for q in kept_q_dims}
    else:
        # Use user-provided weights, then normalize to sum to 1
        total_w = sum(q_weights.get(q, 0.0) for q in kept_q_dims)
        if total_w <= 0:
            # If the total weight is not positive, revert to equal weights
            weight_each = 1.0 / len(kept_q_dims)
            weights = {q: weight_each for q in kept_q_dims}
        else:
            weights = {q: q_weights.get(q, 0.0) / total_w for q in kept_q_dims}

    content_value_score = 0.0
    for q in kept_q_dims:
        content_value_score += weights[q] * q_avg_dict.get(q, 0.0)

    content_value_score = round(content_value_score, 1)
    return q_avg_dict, content_value_score


def process_single_record(
    record: Dict[str, Any],
    all_matched_char_ids: Set[str],
) -> Dict[str, Any]:
    """
    Processing a single "evolved_claim" record:
    - Filter the propagation nodes based on role attributes
    - Filter by Q dimension (for content value score)
    - Calculate:
        * Quantified propagation depth
        * Quantified propagation size
        * Propagation Ability score
        * Average value of Q dimension
        * Content value score
        * Final "check-worthiness" label
    """
    final_claim = record.get("final_claim")

    # 1. Extracting Nodes
    all_nodes = extract_nodes_from_record(record)

    # 2. In the propagation path, only the roles that are permitted after attribute filtering are retained.
    filtered_nodes = filter_nodes_by_characters(all_nodes, all_matched_char_ids)

    # 3. Calculate relevant indicators
    depth_score, scale_score, propagation_ability_score = compute_propagation_metrics(
        filtered_nodes=filtered_nodes,
        all_matched_char_ids=all_matched_char_ids,
        max_depth=MAX_PROPAGATION_DEPTH,
        depth_weight=DEPTH_SCORE_WEIGHT,
        scale_weight=SCALE_SCORE_WEIGHT
    )

    # 4. Calculate the content value score (only considering KEPT_Q_DIMENSIONS)
    q_avg_dict, content_value_score = compute_content_value_score(
        filtered_nodes=filtered_nodes,
        kept_q_dims=KEPT_Q_DIMENSIONS,
        q_weights=Q_DIMENSION_WEIGHTS
    )

    # 5. Final check-worthiness label:
    check_worthiness_label = round(
        PROPAGATION_VS_CONTENT_WEIGHT * propagation_ability_score +
        (1 - PROPAGATION_VS_CONTENT_WEIGHT) * content_value_score,
        1
    )

    # 6. Output Structure: Only includes evolved_claim + each indicator
    result = {
        "final_claim": final_claim,
        # Related to Propagation Ability
        "propagation_depth_score": depth_score,
        "propagation_scale_score": scale_score,
        "propagation_ability_score": propagation_ability_score,
        # Related to Content Value
        "content_dimension_averages": q_avg_dict,
        "content_value_score": content_value_score,
        # Final label
        "check_worthiness_label": check_worthiness_label,
    }

    return result


def main():
    characters = load_characters(CHARACTERS_FILE)

    all_matched_char_ids = get_characters_by_attributes(characters, ATTRIBUTE_FILTERS)

    records = load_input_records(INPUT_FILE)

    results = []
    for record in records:
        result = process_single_record(record, all_matched_char_ids)
        results.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Processing complete. {len(records)} records processed. Results written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()