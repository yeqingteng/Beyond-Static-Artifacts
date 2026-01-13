import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from openai import OpenAI


def build_client(api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://example.org")
    if not api_key:
        raise ValueError("Missing API key. Provide --api-key or set OPENAI_API_KEY.")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm_with_retry(
    client: OpenAI,
    model: str,
    system_text: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_base_delay: int,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[API ERROR] attempt {attempt}/{max_retries}: {e}", file=sys.stderr)
            if attempt == max_retries:
                return ""
            delay = retry_base_delay * (2 ** (attempt - 1))
            time.sleep(delay)
    return ""


def safe_json_loads_object(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    try:
        data = json.loads(raw_text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        data = json.loads(raw_text[start:end])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")



def build_seed_prompt(topic_domain: str, title: str, content: str) -> str:
    return f"""
You are a careful factual claim generator.

The article belongs to the following high-level topic domain:
{topic_domain}

Your task:
- Carefully read the title and content provided below.
- Based only on the information in the title and content, generate exactly ONE concise factual claim.
- The claim MUST:
  - be a single declarative sentence
  - be fully grounded in the article (no speculation or hallucination)
  - explicitly include:
    (1) subject information,
    (2) location information (where the event happens),
    (3) time information (when the event happens).
- If the content does not contain location information or time information, then the claim does not need to include that type of information.

Output format (IMPORTANT):
- Output MUST be strict JSON only, with key: "seed_claim".
- "seed_claim" must be non-empty, single sentence, no newline.
- Do not output any extra text outside the JSON.

Narrative (claim title):
{title}

Abstract (claim content):
{content}

Return JSON only.
""".strip()


def build_nonclaim_prompt(refer_content: str) -> str:
    return f"""
You are generating ONE "non-claim" sentence based on a given passage.

Definition of non-claim (choose exactly ONE type):
1) Not a claim (non-assertive utterances):
Questions, commands, exclamations, imperative sentence.

2) Personal experience:
Statements about a speaker's own experiences, feelings, abilities, or internal states that cannot be independently verified using publicly available evidence.

PASSAGE:
\"\"\"{refer_content}\"\"\"

Task:
- Generate exactly ONE non-claim sentence that is clearly related to the passage topic/content.
- Then assign "type" as exactly one of:
  - "Not a claim (non-assertive utterances)"
  - "Personal experience"
- Output MUST be strict JSON only, with keys: "claim", "type".
- "claim" must be non-empty, single sentence, no newline.
- Do not output any extra text outside the JSON.

Return JSON only.
""".strip()


def parse_seed_object(raw: str) -> Optional[str]:
    obj = safe_json_loads_object(raw)
    if obj is None:
        return None
    c = str(obj.get("seed_claim", "")).strip()
    if not c:
        # allow fallback key if model uses "claim"
        c = str(obj.get("claim", "")).strip()
    if not c:
        return None
    return c.replace("\n", " ").strip()


def parse_nonclaim(raw: str) -> Optional[str]:
    obj = safe_json_loads_object(raw)
    if obj is None:
        return None
    c = str(obj.get("claim", "")).strip()
    if not c:
        return None
    return c.replace("\n", " ").strip()


def run_generate_seed_and_non_from_article(
    client: OpenAI,
    model_seed: str,
    model_non: str,
    input_json: str,
    seed_jsonl: str,
    non_jsonl: str,
    num_seed: int,
    num_non: int,
    max_retries: int,
    retry_base_delay: int,
    temperature_seed: float,
    temperature_non: float,
    max_tokens_seed: int,
    max_tokens_non: int,
) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object: {topic: [articles...], ...}")

    system_seed = "You are a careful factual claim generator. Output must be a JSON object only."
    system_non = "You generate a related non-claim. Output must be a JSON object only."

    seed_id = 1
    non_id = 1

    all_articles = []
    for topic, items in data.items():
        if isinstance(items, list):
            for it in items:
                all_articles.append((topic, it))

    with open(seed_jsonl, "w", encoding="utf-8") as f_seed, open(non_jsonl, "w", encoding="utf-8") as f_non:
        for topic, item in tqdm(all_articles, desc="Generating seed+non (from article)"):
            refer_title = str(item.get("title", "")).strip()
            refer_content = str(item.get("content", "")).strip()

            # ---------- seed-claims from article (num_seed calls, 1 each) ----------
            if num_seed > 0:
                seed_prompt = build_seed_prompt(topic, refer_title, refer_content)
                for _ in range(num_seed):
                    raw_seed = call_llm_with_retry(
                        client=client,
                        model=model_seed,
                        system_text=system_seed,
                        user_text=seed_prompt,
                        temperature=temperature_seed,
                        max_tokens=max_tokens_seed,
                        max_retries=max_retries,
                        retry_base_delay=retry_base_delay,
                    )
                    sc = parse_seed_object(raw_seed) or ""
                    write_jsonl_line(
                        f_seed,
                        {
                            "id": seed_id,
                            "seed_claim": sc,
                            "topic": topic,
                            "refer_title": refer_title,
                            "refer_content": refer_content,
                        },
                    )
                    seed_id += 1

            # ---------- non-claims from article (num_non calls, 1 each) ----------
            if num_non > 0:
                non_prompt = build_nonclaim_prompt(refer_content)
                for _ in range(num_non):
                    raw_non = call_llm_with_retry(
                        client=client,
                        model=model_non,
                        system_text=system_non,
                        user_text=non_prompt,
                        temperature=temperature_non,
                        max_tokens=max_tokens_non,
                        max_retries=max_retries,
                        retry_base_delay=retry_base_delay,
                    )
                    nc = parse_nonclaim(raw_non) or ""
                    write_jsonl_line(
                        f_non,
                        {
                            "id": non_id,
                            "non_claim": nc,
                            "topic": topic,
                            "refer_title": refer_title,
                            "refer_content": refer_content,
                        },
                    )
                    non_id += 1

    print(f"[DONE] seed jsonl -> {seed_jsonl} (total lines: {seed_id-1})")
    print(f"[DONE] non  jsonl -> {non_jsonl} (total lines: {non_id-1})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSON: {topic: [articles...], ...}")
    ap.add_argument("--seed_out", required=True, help="Output seed jsonl path")
    ap.add_argument("--non_out", required=True, help="Output non jsonl path")

    ap.add_argument("--num_seed", type=int, default=10, help="Seed claims per article")
    ap.add_argument("--num_non", type=int, default=1, help="Non-claims per article")

    ap.add_argument("--model_seed", default="gpt-4o", help="Model for seed claim generation")
    ap.add_argument("--model_non", default="gpt-4o", help="Model for non-claim generation")

    ap.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    ap.add_argument("--base-url", default=None, help="Base URL (or set OPENAI_BASE_URL)")

    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--retry_base_delay", type=int, default=2)

    ap.add_argument("--temperature_seed", type=float, default=0.7)
    ap.add_argument("--temperature_non", type=float, default=0.7)

    ap.add_argument("--max_tokens_seed", type=int, default=256)
    ap.add_argument("--max_tokens_non", type=int, default=256)

    args = ap.parse_args()

    if args.num_seed < 0 or args.num_non < 0:
        raise ValueError("--num_seed and --num_non must be >= 0")

    client = build_client(args.api_key, args.base_url)

    run_generate_seed_and_non_from_article(
        client=client,
        model_seed=args.model_seed,
        model_non=args.model_non,
        input_json=args.input,
        seed_jsonl=args.seed_out,
        non_jsonl=args.non_out,
        num_seed=args.num_seed,
        num_non=args.num_non,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        temperature_seed=args.temperature_seed,
        temperature_non=args.temperature_non,
        max_tokens_seed=args.max_tokens_seed,
        max_tokens_non=args.max_tokens_non,
    )


if __name__ == "__main__":
    main()

