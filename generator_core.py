#!/usr/bin/env python3
"""
generator_core.py
-----------------
Shared core for grouped dataset generators. Group-specific scripts import from
here so the 700 lines of generation logic isn't duplicated.

Each group script needs to define:
    GROUP_NAME: str                        # e.g. "system_device_info"
    GROUP_TOOL_ALLOWLIST: set[str]         # which tools from the full catalog belong to this group
    GROUP_SEEDS: list[dict]                # scenarios for this group only
    DEFAULT_OUTPUT: str                    # default output filename

then call:
    from generator_core import run_group
    run_group(GROUP_NAME, GROUP_TOOL_ALLOWLIST, GROUP_SEEDS, DEFAULT_OUTPUT)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a synthetic dataset generator. Your job is to produce ONE high-quality tool-calling sample for an Android assistant.

⚠️ CRITICAL: You MUST strictly follow the Intent given below. Do NOT drift to an unrelated scenario.

The output MUST be valid JSON matching this EXACT format:

{
  "scenario": "<one-sentence description of the situation>",
  "expected_outcome": "<one-sentence description of what the assistant accomplishes>",
  "chatbot_role": "<short role description>",
  "turns": [
    {"role": "user", "content": "<user's full self-contained request — includes time, name, value, etc. — no missing info>"},
    {
      "role": "assistant",
      "content": "<brief reasoning, e.g. 'Checking your battery level now.'>",
      "tools_called": [
        {
          "name": "<exact tool name from AVAILABLE TOOLS>",
          "description": "<brief paraphrase of what the tool does>",
          "input_parameters": { <arguments as JSON OBJECT, NOT a string> },
          "output": "<realistic result as a JSON-encoded string>"
        }
      ]
    },
    {"role": "assistant", "content": "<natural-language summary that REFERENCES the actual values returned by the tool — e.g. 'Your battery is at 23%. I'd suggest charging soon.' — NO tools_called here>"}
  ]
}

HARD REQUIREMENTS:
1. EXACTLY 3 turns. NOT 2, NOT 4. Three turns total: user → assistant (with tool calls) → assistant (summary, NO tools_called).
2. The user's single message contains EVERYTHING needed (specific time, contact, value). The assistant does NOT ask back.
3. The FIRST assistant turn MUST contain `tools_called` with at least 1 tool call.
4. The LAST assistant turn MUST NOT contain `tools_called`. Its content is a natural-language summary that references the actual values from the tool output (don't invent new ones).
5. Do NOT add a 4th turn (no follow-up user, no extra assistant). The summary in turn 3 ends the conversation.
6. role is ONLY "user" or "assistant". NEVER "system", NEVER "tool".
7. Every tool name MUST appear in AVAILABLE TOOLS.
8. `input_parameters` is a JSON OBJECT (dict), NOT a string.
9. `output` is a JSON-encoded STRING simulating what the tool would return.
10. Required parameters MUST be present; do NOT include keys not in the tool's schema.
11. Use the Target tools; do NOT substitute unrelated tools.
12. For inter-group scenarios: tools from EACH listed group must be called. Parallel calls in the same first-assistant turn are fine.

VARIETY:
- Vary the user's opening phrasing. Do NOT always start with "My", "Can you", "Please".
- Vary the summary's opening — sometimes confirm action, sometimes lead with the value.
- Vary argument values — use realistic but DIFFERENT specifics each time.

Output ONLY the JSON object. No prose, no markdown fences.
"""


def build_user_prompt(tools_block: str, seed: dict[str, Any]) -> str:
    target = seed.get("target_tools")
    target_line = (f"- **Target tools (MUST use at least 60% of these)**: {', '.join(target)}"
                   if target else
                   "- Target tools: choose 1-3 from AVAILABLE TOOLS.")
    expected = seed.get("expected_outcome", "Assistant resolves the user's request.")

    min_tc = seed.get("min_tool_calls", 1)
    groups_req = seed.get("groups_required") or []
    multi_group_line = (
        f"- Inter-group: tools called MUST span these groups: {', '.join(groups_req)}. "
        f"Use at least one tool from EACH group (parallel calls in the same assistant turn)."
        if groups_req else ""
    )

    variety_hints = [
        "Use short, colloquial phrasing for the user.",
        "Use formal phrasing for the user.",
        "The user's message is mid-conversation (no greeting).",
        "User phrases it as a question.",
        "User phrases it as a direct command.",
        "User sounds impatient.",
        "User is curious and asks why.",
    ]
    variety = random.choice(variety_hints)

    return f"""AVAILABLE TOOLS:
{tools_block}

SCENARIO (match this exactly — do not drift to a different topic):
- **Intent**: {seed['intent']}
{target_line}
- Strategy hint: {seed['hint']}
- User persona: {seed['persona']}
- Expected outcome: {expected}
- Minimum tool calls: {min_tc}
{multi_group_line}
- Variety nudge: {variety}

The user's SINGLE message must be self-contained — include all specifics (time, name, value) so the assistant doesn't need to ask back.
The assistant responds in TWO consecutive turns:
  (a) first assistant turn carries the tool call(s) with brief reasoning,
  (b) second assistant turn is a natural-language summary that references the actual values from the tool output (no new info).

Generate the sample now. Output ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Tool catalog filtering — per-group
# ---------------------------------------------------------------------------

def load_tools_filtered(path: Path, allowlist: set[str]):
    """Load full catalog, keep only tools in allowlist. Return (list, schema_index)."""
    with path.open() as f:
        raw = json.load(f)
    tool_list, index = [], {}
    for entry in raw:
        fn = entry["function"]
        if fn["name"] not in allowlist:
            continue
        tool_list.append(entry)
        index[fn["name"]] = fn["parameters"]
    return tool_list, index


def compact_tools_for_prompt(tools):
    return json.dumps([{
        "name": t["function"]["name"],
        "description": t["function"].get("description", ""),
        "parameters": t["function"]["parameters"],
    } for t in tools], indent=2)


# ---------------------------------------------------------------------------
# Seed prep
# ---------------------------------------------------------------------------

def prepare_seeds(handcrafted, schema_index, tools, verbose=True):
    """Drop seeds referencing tools outside the group allowlist; auto-cover uncovered."""
    available = set(schema_index.keys())
    kept, dropped = [], []
    for s in handcrafted:
        unknown = [t for t in (s.get("target_tools") or []) if t not in available]
        if unknown:
            dropped.append((s["intent"][:50], unknown))
            continue
        kept.append(s)

    if verbose and dropped:
        print(f"[!] Dropped {len(dropped)} seeds referencing out-of-group tools:",
              flush=True)
        for intent, unk in dropped[:3]:
            print(f"    - {intent!r} → unknown {unk}", flush=True)

    covered = set()
    for s in kept:
        for t in s.get("target_tools") or []:
            covered.add(t)
    uncovered = sorted(available - covered)
    desc_by_name = {t["function"]["name"]: t["function"].get("description", "") for t in tools}
    for name in uncovered:
        kept.append({
            "intent": f"User request requiring the '{name}' tool. Purpose: {desc_by_name.get(name,'')[:140]}",
            "target_tools": [name],
            "hint": f"Use '{name}' with valid arguments per schema.",
            "persona": "casual",
            "expected_outcome": f"Request resolved using {name}.",
        })

    if verbose:
        print(f"[+] Seeds in group: {len(kept)} ({len(uncovered)} auto-coverage)",
              flush=True)
    return kept


# ---------------------------------------------------------------------------
# Parsing & validation
# ---------------------------------------------------------------------------

JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate_sample(sample, schema_index, target_tools=None, strict_targets=True,
                    group_allowlist=None, seed=None, tool_to_group=None):
    """
    Validate a generated sample against the schema and the strict rules.

    `seed` (optional): if provided, per-seed thresholds override defaults:
        requires_clarification (default True)
        min_tool_calls         (default 2)
        min_turns              (default 6)
        groups_required        (default None — only for inter-group seeds)

    `tool_to_group` (optional): map tool_name -> group_name. Required only
    when `seed` has a non-empty `groups_required` (inter-group runs).
    """
    seed = seed or {}
    requires_clar = seed.get("requires_clarification", False)   # no clarification
    min_tool_calls = int(seed.get("min_tool_calls", 1))         # ≥1 tool call
    min_turns_req  = int(seed.get("min_turns", 3))              # 3-turn default
    max_turns_req  = int(seed.get("max_turns", min_turns_req))  # strict upper bound (default = min_turns)
    groups_req     = set(seed.get("groups_required") or [])

    # Detect mode for conditional rules below.
    is_multi_turn = (min_turns_req >= 3) or requires_clar

    # ---- shape ----
    if not isinstance(sample, dict) or "turns" not in sample:
        return False, "missing 'turns'"
    turns = sample["turns"]
    if not isinstance(turns, list) or len(turns) < 2:
        return False, "too few turns"
    if turns[0].get("role") != "user":
        return False, "first turn must be 'user'"
    if turns[-1].get("role") != "assistant":
        return False, "last turn must be 'assistant'"
    # Only enforce "last turn has no tool calls" in multi-turn mode (where the
    # final turn is the natural-language summary). In single-turn mode the
    # only assistant turn IS the tool-call turn.
    if is_multi_turn and turns[-1].get("tools_called"):
        return False, "last assistant turn must not have tools_called (multi-turn mode)"

    # ---- min/max turns ----
    if len(turns) < min_turns_req:
        return False, f"only {len(turns)} turns (need >={min_turns_req})"
    if len(turns) > max_turns_req:
        return False, f"{len(turns)} turns (max allowed {max_turns_req})"

    # ---- per-turn validation + collect tool calls ----
    n_tool_calls = 0
    tools_called = []
    user_count = 0
    first_assistant_idx = None
    first_call_turn = None

    for i, t in enumerate(turns):
        role = t.get("role")
        if role not in ("user", "assistant"):
            return False, f"turn {i}: invalid role '{role}'"
        if not isinstance(t.get("content"), str):
            return False, f"turn {i}: content not string"
        if role == "user":
            user_count += 1
            if t.get("tools_called"):
                return False, f"turn {i}: user turn has tools_called"
        if role == "assistant":
            if first_assistant_idx is None:
                first_assistant_idx = i
            tc_list = t.get("tools_called") or []
            if tc_list and first_call_turn is None:
                first_call_turn = i
            for j, tc in enumerate(tc_list):
                n_tool_calls += 1
                name = tc.get("name")
                if name not in schema_index:
                    return False, f"turn {i} tc{j}: unknown tool '{name}' (out of group)"
                tools_called.append(name)
                args = tc.get("input_parameters")
                if not isinstance(args, dict):
                    return False, f"turn {i} tc{j}: input_parameters not dict"
                if not isinstance(tc.get("output"), str):
                    return False, f"turn {i} tc{j}: output not string"
                params = schema_index[name]
                missing = [k for k in params.get("required", []) if k not in args]
                if missing:
                    return False, f"turn {i} tc{j} '{name}': missing required {missing}"
                allowed = set((params.get("properties") or {}).keys())
                unknown = [k for k in args if k not in allowed]
                if unknown:
                    return False, f"turn {i} tc{j} '{name}': unknown args {unknown}"

    # ---- ≥2 user turns (only when clarification flow is required) ----
    if requires_clar and user_count < 2:
        return False, f"only {user_count} user turn(s) (need >=2 with clarification)"

    # ---- multi-tool call requirement ----
    if n_tool_calls < min_tool_calls:
        return False, f"only {n_tool_calls} tool call(s) (need >={min_tool_calls})"
    if n_tool_calls > 10:
        return False, f"too many tool_calls ({n_tool_calls})"

    # ---- clarification turn ----
    if requires_clar:
        if first_call_turn is None:
            return False, "no tool calls present"
        if first_assistant_idx is None or first_assistant_idx >= first_call_turn:
            return False, "first assistant turn already has tool calls (no clarification)"
        first_asst = turns[first_assistant_idx]
        if first_asst.get("tools_called"):
            return False, "clarification turn has tool_calls"
        clar_content = (first_asst.get("content") or "").strip()
        if "?" not in clar_content and len(clar_content) < 25:
            return False, "clarification turn doesn't look like a question"

    # ---- target tools ----
    if strict_targets and target_tools:
        target_set = set(target_tools)
        called_set = set(tools_called)
        hits = called_set & target_set
        if not hits:
            return False, f"no target tools called (wanted {sorted(target_set)}, got {sorted(called_set)})"
        if len(target_set) >= 2 and len(hits) / len(target_set) < 0.6:
            return False, f"only {len(hits)}/{len(target_set)} target tools used"
        if len(called_set - target_set) > 2:
            return False, f"too many off-target tools: {sorted(called_set - target_set)}"

    # ---- inter-group coverage ----
    if groups_req:
        if tool_to_group is None:
            return False, "tool_to_group required for inter-group validation"
        groups_seen = {tool_to_group.get(n) for n in tools_called}
        groups_seen.discard(None)
        missing_groups = groups_req - groups_seen
        if missing_groups:
            return False, f"groups not covered: {sorted(missing_groups)}"
        if len(groups_seen) < 2:
            return False, "only 1 group hit (need >=2 for inter-group)"

    return True, "ok"


# ---------------------------------------------------------------------------
# Model loading (HuggingFace, no vLLM)
# ---------------------------------------------------------------------------

def load_model_hf(model_name, device, dtype, load_in_4bit, load_in_8bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # detect local folder vs HF Hub ID
    model_path = Path(model_name).expanduser()
    is_local = model_path.exists() and model_path.is_dir()
    if is_local:
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Local model folder '{model_path}' has no config.json. "
                f"Point --model at the directory containing config.json, "
                f"tokenizer.json, and *.safetensors files."
            )
        resolved = str(model_path.resolve())
        print(f"[+] Loading LOCAL model from: {resolved}", flush=True)
        load_kwargs_extra = {"local_files_only": True}
    else:
        resolved = model_name
        print(f"[+] Loading from HuggingFace Hub: {model_name}", flush=True)
        load_kwargs_extra = {}

    tok = AutoTokenizer.from_pretrained(
        resolved, trust_remote_code=True, **load_kwargs_extra,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = {"trust_remote_code": True, **load_kwargs_extra}
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}.get(dtype, torch.bfloat16)
        kwargs["torch_dtype"] = torch_dtype
        if device == "auto":
            kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
    if not (load_in_4bit or load_in_8bit) and device != "auto":
        model = model.to(device)
    model.eval()
    return model, tok


def build_chat_prompt(tokenizer, system, user, enable_thinking=False):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)


def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, top_p, device):
    import torch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=4096)
    if device == "auto":
        emb_device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(emb_device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# The main generation loop — used by every group script
# ---------------------------------------------------------------------------

def _normalize(text):
    t = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def run_group_generation(
    group_name: str,
    group_allowlist: set[str],
    group_seeds: list[dict],
    tools_file: Path,
    output_file: Path,
    num_samples: int,
    model,  # pre-loaded HF model
    tokenizer,
    batch_size: int = 4,
    temperature: float = 0.85,
    top_p: float = 0.95,
    max_new_tokens: int = 2048,
    enable_thinking: bool = False,
    device: str = "auto",
    debug_dir: Path | None = None,
    strict_targets: bool = True,
    tool_to_group: dict[str, str] | None = None,   # NEW: only set for inter-group runs
) -> int:
    """Run generation for one tool group. Returns count of valid samples written."""
    tools, schema_index = load_tools_filtered(tools_file, group_allowlist)
    if not tools:
        print(f"[!] Group '{group_name}': no tools from allowlist found in {tools_file}",
              flush=True)
        return 0
    print(f"[+] Group '{group_name}': {len(tools)} tools "
          f"({list(schema_index.keys())})", flush=True)

    tools_block = compact_tools_for_prompt(tools)
    active_seeds = prepare_seeds(group_seeds, schema_index, tools)
    if not active_seeds:
        print(f"[!] Group '{group_name}': no usable seeds after filtering", flush=True)
        return 0

    # Resume
    completed_ids = set()
    accepted_sigs = set()
    seed_counts = Counter()
    if output_file.exists():
        with output_file.open() as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    completed_ids.add(obj.get("id", ""))
                    turns = obj.get("turns", [])
                    if turns and turns[0].get("role") == "user":
                        accepted_sigs.add(_normalize(turns[0]["content"])[:200])
                    seed_counts[obj.get("scenario_intent", "")] += 1
                except json.JSONDecodeError:
                    pass
        if completed_ids:
            print(f"[+] Resuming '{group_name}': {len(completed_ids)} samples existing",
                  flush=True)
    if len(completed_ids) >= num_samples:
        print(f"[+] Group '{group_name}' already has {len(completed_ids)} samples. Skipping.",
              flush=True)
        return len(completed_ids)

    # Per-seed cap: 1.5x ideal share, floor 3
    ideal = max(1, num_samples / len(active_seeds))
    per_seed_cap = max(3, int(ideal * 2.0))
    print(f"[+] Balance: per_seed_cap={per_seed_cap} ({len(active_seeds)} seeds)",
          flush=True)

    # Shuffled seed queue
    seed_q = list(active_seeds)
    random.shuffle(seed_q)
    cursor = [0]  # mutable from inner fn

    def next_seed():
        tries = 0
        while tries < len(seed_q) * 2:
            if cursor[0] >= len(seed_q):
                random.shuffle(seed_q)
                cursor[0] = 0
            s = seed_q[cursor[0]]
            cursor[0] += 1
            tries += 1
            if seed_counts[s["intent"]] >= per_seed_cap:
                continue
            return s
        return seed_q[cursor[0] % len(seed_q)]

    n_valid = len(completed_ids)
    n_attempts = 0
    max_attempts = num_samples * 6

    if debug_dir:
        debug_dir = Path(debug_dir) / group_name
        debug_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=num_samples, initial=n_valid, desc=f"[{group_name}]")
    out_fh = output_file.open("a")
    try:
        while n_valid < num_samples and n_attempts < max_attempts:
            this_batch = min(batch_size, num_samples - n_valid)
            prompts, seeds_used = [], []
            for _ in range(this_batch):
                s = next_seed()
                seeds_used.append(s)
                up = build_user_prompt(tools_block, s)
                prompts.append(build_chat_prompt(tokenizer, SYSTEM_PROMPT, up,
                                                  enable_thinking=enable_thinking))
            t0 = time.time()
            texts = generate_batch(model, tokenizer, prompts, max_new_tokens,
                                    temperature, top_p, device)
            dt = time.time() - t0

            batch_valid = 0
            batch_rej = Counter()
            for text, s in zip(texts, seeds_used):
                n_attempts += 1
                text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
                text = re.sub(r"^[\s\S]*?</think>\s*", "", text).strip()
                obj = extract_json(text)
                if obj is None:
                    batch_rej["no_json"] += 1
                    if debug_dir and n_attempts <= 3:
                        (debug_dir / f"reject_{n_attempts}_no_json.txt").write_text(
                            f"=== SEED ===\n{s['intent']}\n\n=== RAW ===\n{text[:4000]}")
                    continue
                ok, reason = validate_sample(
                    obj, schema_index,
                    target_tools=s.get("target_tools"),
                    strict_targets=strict_targets,
                    group_allowlist=group_allowlist,
                    seed=s,                          # enables per-seed thresholds
                    tool_to_group=tool_to_group,     # enables inter-group coverage check
                )
                if not ok:
                    batch_rej[reason] += 1
                    if debug_dir and n_attempts <= 5:
                        safe = reason[:30].replace(" ", "_").replace("/", "_")
                        (debug_dir / f"reject_{n_attempts}_{safe}.json").write_text(
                            json.dumps(obj, indent=2)[:4000])
                    continue
                # Dup check
                first_user = next((t["content"] for t in obj["turns"]
                                   if t.get("role") == "user"), "")
                sig = _normalize(first_user)[:200]
                if sig and sig in accepted_sigs:
                    batch_rej["duplicate"] += 1
                    continue
                if seed_counts[s["intent"]] >= per_seed_cap:
                    batch_rej["seed_cap"] += 1
                    continue
                accepted_sigs.add(sig)
                seed_counts[s["intent"]] += 1

                record = {
                    "id": f"sample_{group_name}_{uuid.uuid4().hex[:8]}",
                    "group": group_name,
                    "scenario": obj.get("scenario", s["intent"]),
                    "expected_outcome": obj.get("expected_outcome",
                                                s.get("expected_outcome", "")),
                    "chatbot_role": obj.get("chatbot_role",
                        f"Android on-device assistant specializing in {group_name} tools."),
                    "scenario_intent": s["intent"],
                    "scenario_persona": s["persona"],
                    "target_tools": s.get("target_tools", []),
                    "turns": obj["turns"],
                }
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_valid += 1
                batch_valid += 1
                pbar.update(1)
                if n_valid >= num_samples:
                    break

            top_rej = dict(batch_rej.most_common(3))
            print(f"[{group_name}][batch] attempts={n_attempts} "
                  f"valid={n_valid}/{num_samples} batch={batch_valid}/{this_batch} "
                  f"sec={dt:.1f} rej={top_rej}", flush=True)
    finally:
        out_fh.close()
        pbar.close()

    print(f"[+] Group '{group_name}' done: {n_valid} samples written to {output_file}",
          flush=True)
    return n_valid


# ---------------------------------------------------------------------------
# CLI entry — each group script calls this
# ---------------------------------------------------------------------------

def run_group(group_name: str, allowlist: set[str], seeds: list[dict],
              default_output: str):
    ap = argparse.ArgumentParser(description=f"Generate {group_name} samples")
    ap.add_argument("--tools-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(default_output))
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="debug_rejects")
    ap.add_argument("--no-strict-targets", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    print(f"[+] Loading model {args.model}...", flush=True)
    model, tokenizer = load_model_hf(
        args.model, args.device, args.dtype, args.load_in_4bit, args.load_in_8bit,
    )
    print(f"[+] Model ready.", flush=True)

    run_group_generation(
        group_name=group_name,
        group_allowlist=allowlist,
        group_seeds=seeds,
        tools_file=args.tools_file,
        output_file=args.output,
        num_samples=args.num_samples,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        device=args.device,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        strict_targets=not args.no_strict_targets,
    )


# ---------------------------------------------------------------------------
# Inter-group entry point — used by gen_12_inter_group.py
# ---------------------------------------------------------------------------

def run_intergroup(
    group_name: str,
    allowlists_by_group: dict[str, set[str]],
    seeds: list[dict],
    default_output: str,
):
    """Like run_group(), but for cross-group scenarios."""
    combined_allowlist: set[str] = set().union(*allowlists_by_group.values())
    tool_to_group: dict[str, str] = {}
    for grp, tools in allowlists_by_group.items():
        for t in tools:
            tool_to_group[t] = grp

    valid_groups = set(allowlists_by_group.keys())
    cleaned: list[dict] = []
    for s in seeds:
        gr = set(s.get("groups_required") or [])
        if len(gr) < 2:
            print(f"[!] inter-group seed without >=2 groups_required, skipping: "
                  f"{s.get('intent','')[:60]!r}", flush=True)
            continue
        unknown_grp = gr - valid_groups
        if unknown_grp:
            print(f"[!] seed references unknown groups {unknown_grp}, "
                  f"skipping: {s.get('intent','')[:60]!r}", flush=True)
            continue
        unknown_tools = [t for t in (s.get("target_tools") or [])
                         if t not in combined_allowlist]
        if unknown_tools:
            print(f"[!] seed targets tools not in combined allowlist "
                  f"{unknown_tools}, skipping: {s.get('intent','')[:60]!r}",
                  flush=True)
            continue
        cleaned.append(s)

    if not cleaned:
        print(f"[!] No usable inter-group seeds after sanity check.", flush=True)
        return

    ap = argparse.ArgumentParser(description=f"Generate {group_name} samples")
    ap.add_argument("--tools-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(default_output))
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--debug-dir", type=str, default="debug_rejects")
    ap.add_argument("--no-strict-targets", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    print(f"[+] Loading model {args.model}...", flush=True)
    model, tokenizer = load_model_hf(
        args.model, args.device, args.dtype, args.load_in_4bit, args.load_in_8bit,
    )
    print(f"[+] Model ready.", flush=True)
    print(f"[+] Inter-group '{group_name}': "
          f"{len(allowlists_by_group)} groups, "
          f"{len(combined_allowlist)} tools combined, "
          f"{len(cleaned)} seeds", flush=True)

    run_group_generation(
        group_name=group_name,
        group_allowlist=combined_allowlist,
        group_seeds=cleaned,
        tools_file=args.tools_file,
        output_file=args.output,
        num_samples=args.num_samples,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        device=args.device,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
        strict_targets=not args.no_strict_targets,
        tool_to_group=tool_to_group,
    )
