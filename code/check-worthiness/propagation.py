import json
import random
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI

CHARACTERS_FILE = "characters_group.json"              
NETWORK_FILE = "dunbar_network.json"  
CLAIMS_FILE = "intervened_propagation_evolved_tpc.jsonl"      
OUTPUT_FILE = "intervened_propagation_evolved_results.json"         

MODEL = "gpt-4o"   
API_KEY = "xxxxxx"  
BASE_URL = "https://example.org"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

MAX_RETRIES = 3          
RETRY_BASE_DELAY = 2     

def call_llm(prompt_text: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "You MUST strictly follow the output format requested by the user."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt_text,
                    },
                ],
                temperature=0.7,
                max_tokens=512,
            )

            content = resp.choices[0].message.content or ""
            return content.strip()

        except Exception as e:
            print(
                f"[API ERROR] attempt {attempt}/{MAX_RETRIES} failed: {e}",
                file=sys.stderr,
            )
            if attempt == MAX_RETRIES:
                return ""
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"[RETRY] wait {delay} seconds before next attempt...", file=sys.stderr)
            time.sleep(delay)



def load_characters(path: str) -> Dict[str, Dict[str, Any]]:
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_network(path: str, all_node_ids: List[str]) -> Tuple[Dict[str, Dict[str, List[str]]], List[str]]:
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_set = set(all_node_ids)
    relations: Dict[str, Dict[str, List[str]]] = {}

    for cid in all_node_ids:
        info = data.get(cid, {}) or {}

        strong_ids = [x.get("id") for x in info.get("strong", []) if isinstance(x, dict) and x.get("id")]
        moderate_ids = [x.get("id") for x in info.get("moderate", []) if isinstance(x, dict) and x.get("id")]

        def _dedup_keep_order(ids: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in ids:
                if x in all_set and x != cid and x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        strong_ids = _dedup_keep_order(strong_ids)
        moderate_ids = _dedup_keep_order(moderate_ids)

        sm_set = set(strong_ids) | set(moderate_ids)
        weak_ids = [nid for nid in all_node_ids if nid != cid and nid not in sm_set]

        relations[cid] = {
            "strong": strong_ids,
            "moderate": moderate_ids,
            "weak": weak_ids,
        }

    return relations, all_node_ids


def build_evaluation_prompt(
        character_id: str,
        character_profile: Dict[str, Any],
        final_claim: str,
        T: int,
        P: int,
        C: int,
) -> str:
    """
    Construct the prompt for the LLM:
    - Introduce the attributes of the character
    - Explain the meanings and current values of T/P/C
    - Provide 6 questions
    - Require the output in JSON format of the 6 scores + whether to continue spreading
    """

    # 将 character 属性整理成文本
    profile_lines = []
    for k, v in character_profile.items():
        profile_lines.append(f"- {k}: {v}")
    profile_text = "\n".join(profile_lines)

    context_text = (
        f"T (Time node) = {T}.\n"
        f"P (Regulatory pressure) = {P}.\n"
        f"C (Trust climate) = {C}.\n\n"
        "Interpretation:\n"
        "- When T = 1, the society is in a sensitive period. "
        "In this case, people tend to consider claims more check-worthy.\n"
        "- When P = 1, regulatory pressure is high, which also increases check-worthiness.\n"
        "- When C = 1, the general trust climate is low, which also increases check-worthiness.\n"
        "You should take these contextual factors into account when judging the check-worthiness.\n"
    )

    question_text = """
You are a virtual user (character) with the following attributes:

{profile_text}

You are now reading the following claim:

[CLAIM START]
{final_claim}
[CLAIM END]

The social context of this claim is:
{context_text}

Please evaluate to what extent this claim is check-worthy from SIX dimensions
and then decide whether you personally want to further spread this claim.

Q1 Relevance to Public Interest
To what extent is this claim directly related to the public interest in your society?
Please output a one-decimal number between 0 and 1 (larger means more check-worthy).

Q2 Controversy / Harm Potential
Could this claim cause public controversy, fear or potential harm?
Please output a one-decimal number between 0 and 1 (larger means more check-worthy).

Q3 Elite person / organization
Does this claim involve influential political actors, public agencies or authoritative organizations?
Please output a one-decimal number between 0 and 1 (larger means more check-worthy).

Q4 Geography or culture Proximity
Is the region or culture mentioned in the claim geographically or socially close to your environment?
Please output a one-decimal number between 0 and 1 (larger means more check-worthy).

Q5 Novelty
Is the information in this claim new or unusual to you?
Please output a one-decimal number between 0 and 1 (larger means more check-worthy).

Q6 Verification Difficulty
From your perspective, is this claim difficult to verify or requires extra information or expert knowledge?
Please output a one-decimal number between 0 and 1 (larger means more check-worthy).

After answering these six questions, you MUST decide:
From your own perspective and attributes, do you want to continue spreading this claim to your social contacts?
If you are willing to spread it, set "propagate" to "yes", otherwise set it to "no".

IMPORTANT OUTPUT INSTRUCTION (Very important requirements for output format):
You MUST output ONLY a single-line JSON object in the following format, with one decimal place for each score:

{{"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0, "Q5": 0.0, "Q6": 0.0, "propagate": "yes"}}

- Use numbers between 0.0 and 1.0 (inclusive) with exactly ONE decimal place.
- "propagate" must be either "yes" or "no".
- Do NOT output any explanation or text outside the JSON.
""".strip()

    return question_text.format(
        profile_text=profile_text,
        final_claim=final_claim,
        context_text=context_text,
    )


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except Exception as e:
        print(f"[PARSE ERROR] cannot extract JSON: {e}", file=sys.stderr)
        return None


def evaluate_claim_for_character(
        character_id: str,
        character_profile: Dict[str, Any],
        final_claim: str,
        T: int,
        P: int,
        C: int,
) -> Dict[str, Any]:
    
    prompt = build_evaluation_prompt(character_id, character_profile, final_claim, T, P, C)
    raw_resp = call_llm(prompt)

    parsed = extract_json_from_text(raw_resp) or {}
    scores = {}
    for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        try:
            val = float(parsed.get(q, 0.0))
        except (ValueError, TypeError):
            val = 0.0
        # Force a decimal number
        scores[q] = round(val, 1)

    propagate_flag = str(parsed.get("propagate", "no")).strip().lower() == "yes"

    return {
        "character_id": character_id,
        "scores": scores,
        "propagate": propagate_flag,
        "raw_output": raw_resp,
    }


# ================== Propagation simulation ==================
MAX_DEPTH = 3        # Maximum Transmission Round
MAX_FANOUT = 3       # Each node can spread to a maximum of 3 neighboring nodes
RANDOM_SEED = 42     

def simulate_propagation_for_claim(
        claim_obj: Dict[str, Any],
        #adjacency: Dict[str, List[str]],
        network_relations: Dict[str, Dict[str, List[str]]],
        node_ids: List[str],
        characters: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    

    final_claim = claim_obj.get("final_claim", "")
    T = int(claim_obj.get("T", 0))
    P = int(claim_obj.get("P", 0))
    C = int(claim_obj.get("C", 0))

    available_nodes = [nid for nid in node_ids if nid in characters]
    if not available_nodes:
        raise ValueError("No overlapping character ids between network and characters.")

    seed = random.choice(available_nodes)

    propagation_log: List[Dict[str, Any]] = []

    queue: List[Tuple[str, int, Optional[str]]] = [(seed, 0, None)]
    visited = set([seed])

    while queue:
        current_id, depth, parent_id = queue.pop(0)

        # Obtain the attributes of this character
        char_profile = characters.get(current_id, {})

        # Call the LLM to obtain the 6-dimensional scores and whether it spreads
        eval_result = evaluate_claim_for_character(
            current_id, char_profile, final_claim, T, P, C
        )

        propagate_flag = eval_result["propagate"]

        # Record the current node
        log_entry = {
            "round": depth,
            "character_id": current_id,
            "parent_id": parent_id,
            "scores": eval_result["scores"],
            "propagate": propagate_flag,
            "children": [],
        }

        # If the spread is allowed and the depth has not reached MAX_DEPTH, then select the neighbors
        children_ids: List[str] = []
        if propagate_flag and depth < MAX_DEPTH:
            rel = network_relations.get(current_id, {"strong": [], "moderate": [], "weak": []})

            selected_children: List[str] = []
            remaining = MAX_FANOUT

            # Select from "strong", "moderate", and "weak" in order of priority, with a maximum of 3 choices.
            for tier in ("strong", "moderate", "weak"):
                if remaining <= 0:
                    break
                candidates = [
                    n for n in rel.get(tier, [])
                    if n not in visited and n in characters
                ]
                if not candidates:
                    continue
                k = min(remaining, len(candidates))
                picked = random.sample(candidates, k=k)
                selected_children.extend(picked)
                remaining -= k

            for child in selected_children:
                visited.add(child)
                queue.append((child, depth + 1, current_id))
                children_ids.append(child)

        log_entry["children"] = children_ids
        propagation_log.append(log_entry)

    return {
        "final_claim": final_claim,
        "T": T,
        "P": P,
        "C": C,
        "seed_character": seed,
        "propagation_log": propagation_log,
    }


# ================== Main process ==================
def main():
    random.seed(RANDOM_SEED)

    print("Loading characters...")
    characters = load_characters(CHARACTERS_FILE)

    print("Loading network...")
    node_ids = list(characters.keys())
    network_relations, node_ids = load_network(NETWORK_FILE, node_ids)

    print("Loading claims...")
    claims: List[Dict[str, Any]] = []
    with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claims.append(json.loads(line))

    print(f"Total claims: {len(claims)}")
    print(f"Saving results to {OUTPUT_FILE} ...")

    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        first = True

        for claim_obj in tqdm(claims, desc="Simulating claims"):
            try:
                result = simulate_propagation_for_claim(
                    claim_obj, network_relations, node_ids, characters
                )

                if not first:
                    out_f.write(",\n")
                first = False

                out_f.write(json.dumps(result, ensure_ascii=False, indent=2))
                out_f.flush()  

            except Exception as e:
                print(f"[ERROR] when simulating one claim: {e}", file=sys.stderr)

        out_f.write("\n]\n")

    print("Done.")

if __name__ == "__main__":
    main()
