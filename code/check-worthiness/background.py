import json
import time
import sys
from typing import Dict, Any
import os
from tqdm import tqdm


INPUT_FILE = "intervened_propagation_evolved.jsonl"
OUTPUT_FILE = "intervened_propagation_evolved_tpc.jsonl"

from openai import OpenAI

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
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt_text
                    },
                ],
                temperature=1.0,
                max_tokens=1024,
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

def build_prompt(final_claim: str) -> str:
    """
    According to a "evolved_claim" text, the LLM needs to output a JSON file that contains only three fields: T, P, and C, with values of 0 or 1.
    """
    prompt = f"""
You will receive a piece of claim and you must classify its social context into three binary variables.

[Text to analyze]
\"\"\"{final_claim}\"\"\"

You must decide values (0 or 1) for the following variables:

(1) Time node T — Sensitive period (1) vs. Normal period (0)
    - T = 1 (sensitive period) if the text is about the outbreak, escalation, or ongoing evolution of a harmful / sensitive event, such as:
        - The CURRENT outbreak or rapid spread of a public health incident
        - A political or social crisis that is happening or escalating
        - A safety accident or major disaster that has just happened or is still unfolding
    - T = 0 (normal period) if the text is general discussion, daily topics, historical review, future outlook, or not clearly tied to an ongoing sensitive event.

(2) Regulatory pressure P — High (1) vs. Low (0)
    - P = 1 (high regulatory pressure) if the text is clearly situated in or directly discussing any of the following:
        - Public health (e.g., epidemic control, infectious disea  ses, major health risks)
        - Politics (elections, legitimacy of government, party struggle, sensitive political events, etc.)
        - National security (war, terrorism, espionage, secession, etc.)
        - Law enforcement / regulation (crackdowns, investigations, bans, fines, suspensions, content removal, etc.)
        - Official policy communication (government announcements, official bulletins, policy notices, legal issues, etc.)
    - P = 0 (low regulatory pressure) otherwise.

(3) Trust climate C — Low trust (1) vs. High/normal trust (0)
    - C = 1 (low trust climate) if the text expresses strong distrust toward "officials / government / institutions / media / experts", including:
        - Strong distrust, accusations of lying or hiding information
        - Conspiracy thinking (e.g., "they are hiding the truth", "there is a secret plot behind it")
        - Claims of cover-up or fabrication (e.g., "officially they are covering it up", "all the data are fake")
    - C = 0 if there is no clear strong distrust, conspiracy, or cover-up narrative.

Your output must be a single valid JSON object with exactly these three keys:
{{
  "T": 0 or 1,
  "P": 0 or 1,
  "C": 0 or 1
}}

Important:
- Output ONLY the JSON object.
- Do NOT add any explanation, comments, or additional text.
"""
    return prompt


def parse_result_to_labels(result_text: str) -> Dict[str, int]:
    try:
        data = json.loads(result_text)

        T = int(data.get("T", 0))
        P = int(data.get("P", 0))
        C = int(data.get("C", 0))

        # Simple Normalization: Only allows 0 or 1
        T = 1 if T == 1 else 0
        P = 1 if P == 1 else 0
        C = 1 if C == 1 else 0

        return {"T": T, "P": P, "C": C}
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output. Raw content: {result_text}. Error: {e}")


# ============ Main process ============
def process_file(input_path: str, output_path: str):
    
    with open(input_path, "r", encoding="utf-8") as fcnt:
        total_lines = sum(1 for line in fcnt if line.strip())

    total = 0    
    success = 0  

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        pbar = tqdm(total=total_lines, desc="Processing final_claim", unit="line")

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                item: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Line {total} is not valid JSON and has been skipped. Content snippet: {line[:100]}...")
                pbar.update(1)
                continue

            final_claim = item.get("final_claim", "")
            if not final_claim:
                print(f"[Warning] Line {total} is missing the 'final_claim' field and has been skipped.")
                pbar.update(1)
                continue

            prompt = build_prompt(final_claim)

            result_text = call_llm(prompt)
            if not result_text:
                print(f"[Warning] Line {total} returned an empty LLM output and has been skipped.")
                pbar.update(1)
                continue

            try:
                labels = parse_result_to_labels(result_text)
            except ValueError as e:
                print(f"[Warning] Failed to parse LLM output on line {total}: {e}")
                pbar.update(1)
                continue

            item.update(labels)

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()  

            success += 1

            pbar.set_postfix_str(f"success={success}, read={total}")
            pbar.update(1)

        pbar.close()

    print(f"Processing finished: {success} succeeded out of {total} total. Output file: {output_path}")


if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)
