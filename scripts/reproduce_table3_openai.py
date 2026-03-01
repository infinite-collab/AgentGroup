"""
Reproduce (approximately) Table 3-style benchmark scores for two OpenAI models:
  - GPT-4 Turbo
  - GPT-3.5 Turbo

This script is intentionally lightweight and self-contained: it renders the same
prompt templates used in this repo (Choose / Update / Converse / Summarize),
calls the OpenAI API, and computes pass rates for:
  Instruction Understanding: CS, BUS, RUS, NoR, NoB
  Environment Understanding: NoA, NoCR, NoC, NoR(resource)

Notes:
  - The exact numbers in the paper depend on the authors' evaluation dataset,
    sampling strategy, and model snapshots. This script is a best-effort
    reproduction harness, not a guaranteed exact match.
  - Fill in API config below before running.

Usage (from `AgentGroup-main/`):
  python scripts/reproduce_table3_openai.py --n-cs 50 --n-update 30 --n-noa 50 --n-nocr 50
  python scripts/reproduce_table3_openai.py --repeats 3 --seed 42 --seed-step 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import requests

# =========================
# OpenAI API CONFIG (EDIT)
# =========================
# Put your API key here (or set env var OPENAI_API_KEY).
OPENAI_API_KEY = ""
# OpenAI API endpoint base URL (default is the public OpenAI endpoint).
# Example: "https://api.openai.com/v1"
OPENAI_BASE_URL = "https://api.zetatechs.com/v1"

# Model names (you can change these to match your account / availability).
OPENAI_GPT35_MODEL = "gpt-3.5-turbo"
OPENAI_GPT4_TURBO_MODEL = "gpt-4-turbo"
OPENAI_GPT52_MODEL = "gpt-5.2"


# =========================
# Prompt / Data Locations
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO_ROOT / "prompt" / "prompt_files"
DEFAULT_SCENARIO_DIR = REPO_ROOT / "storage" / "succession" / "initial_version"


@dataclass(frozen=True)
class CharacterData:
    id_name: str
    name: str
    main_character: bool
    objective: str
    scratch: str
    background: str
    belief: dict[str, int]

    def short_description(self) -> str:
        return self.background

    def main_belief_sentence(self) -> str:
        if not self.belief:
            return ""
        max_score = max(self.belief.values())
        top = [k.strip("。") for k, v in self.belief.items() if v == max_score]
        return "; ".join(top) + "."

    def self_description(self) -> str:
        desc = f"You: {self.id_name}.\n"
        desc += f"Your goal: {self.objective}\n"
        desc += f"Here is your role setting: {self.scratch}\n"
        desc += f"In the public eye, you are: {self.background}\n"
        if self.main_character:
            desc += f"Your thought: {self.main_belief_sentence()} "
        return desc.strip()

    def belief_block(self) -> str:
        # Match Character.get_all_belief() formatting in this repo.
        lines = [f"{k} : {v}" for k, v in self.belief.items()]
        return "\n".join(lines).strip()


@dataclass(frozen=True)
class ResourceData:
    id_number: str
    name: str
    description: str
    influence: int
    owner: str
    topic: str


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_reasoning_effort: str | None = None,
        use_reasoning_fields: bool = True,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._warned_mismatch: set[tuple[str, str]] = set()
        self._default_reasoning_effort = default_reasoning_effort
        self._use_reasoning_fields = use_reasoning_fields

    def chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        timeout_s: int = 60,
        retries: int = 5,
        reasoning_effort: str | None = None,
    ) -> str:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        requested_max_tokens = max_tokens
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        model_lc = model.lower()
        # Gemini-compatible endpoints may spend a large share on reasoning tokens.
        # Reserve more completion budget to avoid truncated visible output.
        if model_lc.startswith("gemini"):
            payload["max_tokens"] = max(payload["max_tokens"], int(max_tokens * 2.5) + 100)
        # Reasoning-heavy models may spend completion budget on hidden reasoning.
        # Nudge them toward concise output and give a safer completion budget floor.
        if self._use_reasoning_fields and model_lc.startswith("gpt-5"):
            effort = reasoning_effort or self._default_reasoning_effort or "minimal"
            payload["reasoning_effort"] = effort
            # High reasoning can consume more completion budget before visible text.
            token_floor = 1800 if effort == "high" else 900
            payload["max_tokens"] = max(payload["max_tokens"], token_floor)
            payload["max_completion_tokens"] = payload["max_tokens"]

        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
                if resp.status_code == 400 and (
                    "reasoning_effort" in payload or "max_completion_tokens" in payload
                ):
                    # Some compatible gateways do not accept these OpenAI-style fields.
                    if "reasoning_effort" in payload:
                        print("[warn] gateway rejected reasoning fields; retrying without reasoning_effort.")
                    payload.pop("reasoning_effort", None)
                    payload.pop("max_completion_tokens", None)
                    continue
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text}")
                resp.raise_for_status()
                data = resp.json()
                returned_model = str(data.get("model", "")).strip()
                if returned_model:
                    key = (model, returned_model)
                    if key not in self._warned_mismatch and model.lower() not in returned_model.lower():
                        # Third-party gateways sometimes remap model names; warn once per pair.
                        print(
                            f"[warn] requested model '{model}' but API returned '{returned_model}'. "
                            "Scores may not be directly comparable."
                        )
                        self._warned_mismatch.add(key)
                choice = data["choices"][0]
                message = choice.get("message") or {}
                content = str(message.get("content", "") or "")
                finish_reason = str(choice.get("finish_reason", "") or "")

                # Auto-retry with larger completion budget when output is truncated.
                if finish_reason == "length":
                    current_max = int(payload.get("max_tokens", requested_max_tokens))
                    max_cap = 8192 if model_lc.startswith("gemini") else 4096
                    if current_max < max_cap:
                        next_max = min(int(current_max * 1.8) + 64, max_cap)
                        payload["max_tokens"] = next_max
                        if "max_completion_tokens" in payload:
                            payload["max_completion_tokens"] = next_max
                        continue
                return content
            except Exception as exc:
                last_error = exc
                sleep_s = min(2**attempt, 20)
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI request failed after {retries} retries: {last_error}")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_prompt(template_rel_path: str, inputs: Sequence[str]) -> str:
    template_path = PROMPT_DIR / template_rel_path
    prompt = _read_text(template_path)
    for idx, value in enumerate(inputs):
        prompt = prompt.replace(f"!<INPUT {idx}>!", str(value))
    marker = "<commentblockmarker>###</commentblockmarker>"
    if marker in prompt:
        prompt = prompt.split(marker, 1)[1]
    return prompt.strip()


def load_characters(characters_dir: Path) -> list[CharacterData]:
    chars: list[CharacterData] = []
    for p in sorted(characters_dir.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        belief_raw = raw.get("belief") or {}
        belief: dict[str, int] = {}
        for k, v in belief_raw.items():
            try:
                belief[str(k)] = int(v)
            except Exception:
                continue
        chars.append(
            CharacterData(
                id_name=str(raw.get("id_name", "")).strip(),
                name=str(raw.get("name", "")).strip(),
                main_character=str(raw.get("main_character", "False")) == "True",
                objective=str(raw.get("objective", "")).strip(),
                scratch=str(raw.get("scratch", "")).strip(),
                background=str(raw.get("background", "")).strip(),
                belief=belief,
            )
        )
    # Sort by ID to keep deterministic ordering (important for strict Action Space checks).
    return sorted(chars, key=lambda c: c.id_name)


def load_resources(resources_dir: Path) -> list[ResourceData]:
    res: list[ResourceData] = []
    for p in sorted(resources_dir.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        try:
            influence = int(raw.get("influence", 0))
        except Exception:
            influence = 0
        res.append(
            ResourceData(
                id_number=str(raw.get("id_number", "")).strip(),
                name=str(raw.get("name", "")).strip(),
                description=str(raw.get("description", "")).strip(),
                influence=influence,
                owner=str(raw.get("owner", "")).strip(),
                topic=str(raw.get("topic", "")).strip(),
            )
        )
    return sorted(res, key=lambda r: r.id_number)


def candidates_block(candidates: Iterable[CharacterData]) -> str:
    # Match how main.py formats candidates: "<ID>: <background>"
    return "\n".join(f"{c.id_name}: {c.short_description()}" for c in candidates).strip()


def synthetic_action_history(lines: int, rng: random.Random) -> str:
    # Keep 1 event per line and avoid blank lines (ground truth counts non-empty lines).
    # Keep every line as a valid event/thought (closer to original repo semantics),
    # while still adding distracting numbers.
    chunks = []
    for i in range(lines):
        a = rng.randint(1, 9999)
        b = rng.randint(1, 9999)
        c = rng.randint(1, 9999)
        chunks.append(
            f"Memory {i + 1}: event-id={a}; round_hint={b % 17}; influence_delta={c % 11}; note=keep negotiating."
        )
    return "\n".join(chunks).strip()


def synthetic_chat_history(rounds: int, rng: random.Random) -> str:
    # Two lines per round (two characters take turns).
    out: list[str] = []
    for i in range(rounds):
        x = rng.randint(10, 999)
        y = rng.randint(10, 999)
        out.append(
            f"C0000 say to C0001: Round {i + 1}, message A, references budget {x} and target {y}."
        )
        out.append(
            f"C0001 say to C0000: Round {i + 1}, message B, rejects plan-{x % 13} and asks for proof {y % 9}."
        )
    return "\n".join(out).strip()


def random_ids(prefix: str, size: int, rng: random.Random, low: int = 1, high: int = 9999) -> list[str]:
    picked = rng.sample(range(low, high + 1), size)
    return [f"{prefix}{v:04d}" for v in picked]


def parse_action_space(text: str) -> list[str] | None:
    # Parse possibly wrapped Action Space block until the next section header.
    m = re.search(
        r"^#+\s*Action Space\s*[:：]\s*(.*?)(?=^\s*#+\s*[A-Za-z][^\n]*[:：]|\Z)",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not m:
        return None
    block = m.group(1).strip()
    ids = re.findall(r"\bC\d{4}\b", block)
    if ids:
        return ids
    parts = [p.strip() for p in re.split(r"[,，\n]+", block)]
    return [p for p in parts if p]


def parse_int_after(label: str, text: str) -> int | None:
    # label is a regex fragment like "Number of Action History" or "Number of Chat Round"
    m = re.search(label + r"\s*[:：]\s*(\d+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_count_flexible(text: str, labels: Sequence[str]) -> int | None:
    # 1) Try strict label extraction with a few aliases.
    for label in labels:
        got = parse_int_after(label, text)
        if got is not None:
            return got
    # 2) Fallback: if model only returns a bare number or short phrase with one number.
    nums = re.findall(r"(\d+)", text)
    if len(nums) == 1:
        try:
            return int(nums[0])
        except Exception:
            return None
    return None


def _parse_delta_list(line: str) -> list[int] | None:
    items = [x.strip() for x in re.split(r"[,，\n]+", line)]
    if not items or (len(items) == 1 and not items[0]):
        return None
    out: list[int] = []
    for it in items:
        if not it:
            return None
        # Allow "C0001:+3", "+3", "+3.", etc.
        if ":" in it:
            it = it.split(":", 1)[1].strip()
        if "：" in it:
            it = it.split("：", 1)[1].strip()
        nums = re.findall(r"[+-]?\d+", it)
        if len(nums) != 1:
            return None
        try:
            out.append(int(nums[0]))
        except Exception:
            return None
    return out


def parse_update_output(text: str) -> tuple[list[int] | None, list[int] | None]:
    rel = None
    bel = None
    m_rel = re.search(
        r"^#+\s*Relationship Change\s*[:：]\s*(.*?)(?=^\s*#+\s*[A-Za-z][^\n]*[:：]|\Z)",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if m_rel:
        rel = _parse_delta_list(m_rel.group(1).strip())
    m_bel = re.search(
        r"^#+\s*Belief Change\s*[:：]\s*(.*?)(?=^\s*#+\s*[A-Za-z][^\n]*[:：]|\Z)",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if m_bel:
        bel = _parse_delta_list(m_bel.group(1).strip())
    return rel, bel


def pct(correct: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * correct / total


def fmt_pct(x: float) -> str:
    return f"{x:.1f}"


@dataclass
class Table3LikeRow:
    model_name: str
    cs: float
    bus: float
    rus: float
    nor_rel_len: float
    nob_bel_len: float
    noa_action_count: float
    nocr_chat_round: float
    noc_character_count: float
    nor_resource_count: float


def average_rows(model_name: str, rows: Sequence[Table3LikeRow]) -> Table3LikeRow:
    if not rows:
        raise ValueError("rows is empty")
    return Table3LikeRow(
        model_name=model_name,
        cs=sum(r.cs for r in rows) / len(rows),
        bus=sum(r.bus for r in rows) / len(rows),
        rus=sum(r.rus for r in rows) / len(rows),
        nor_rel_len=sum(r.nor_rel_len for r in rows) / len(rows),
        nob_bel_len=sum(r.nob_bel_len for r in rows) / len(rows),
        noa_action_count=sum(r.noa_action_count for r in rows) / len(rows),
        nocr_chat_round=sum(r.nocr_chat_round for r in rows) / len(rows),
        noc_character_count=sum(r.noc_character_count for r in rows) / len(rows),
        nor_resource_count=sum(r.nor_resource_count for r in rows) / len(rows),
    )


def benchmark_table3_like(
    *,
    client: OpenAIChatClient,
    model_name: str,
    scenario_dir: Path,
    n_cs: int,
    n_update: int,
    n_noa: int,
    n_nocr: int,
    n_noc: int,
    n_nor_res: int,
    seed: int,
) -> Table3LikeRow:
    rng = random.Random(seed)

    characters = load_characters(scenario_dir / "characters")
    resources = load_resources(scenario_dir / "resources")
    if not characters:
        raise RuntimeError(f"No characters found under: {scenario_dir / 'characters'}")

    # ---------------------------
    # CS: Choose Space
    # ---------------------------
    cs_ok = 0
    cs_total = 0
    for _ in range(n_cs):
        # Use random candidate IDs + richer descriptions to make the action-space extraction harder.
        k = rng.randint(20, 55)
        candidate_ids = random_ids("C", k, rng)
        candidate_desc = "\n".join(
            f"{cid}: Candidate {cid} has strategy-{rng.randint(1, 9)} and resource-index {rng.randint(10, 99)}."
            for cid in candidate_ids
        )
        self_char = characters[0]
        history_for_choose = synthetic_action_history(lines=rng.randint(5, 20), rng=rng)
        prompt = render_prompt(
            "prompt_4_choose.txt",
            [
                self_char.id_name,
                self_char.self_description(),
                "You are in a turn-based debate game.",
                "This is Turn 1.",
                history_for_choose,
                candidate_desc,
                "3",
            ],
        )
        resp = client.chat_completion(model=model_name, prompt=prompt, temperature=0.0, max_tokens=400)
        action_space = parse_action_space(resp)
        cs_total += 1
        if action_space == candidate_ids:
            cs_ok += 1

    # ---------------------------
    # BUS/RUS/NoR/NoB: Update
    # ---------------------------
    max_change = 10
    bus_ok = rus_ok = nor_ok = nob_ok = 0
    update_total = 0
    for _ in range(n_update):
        self_char = rng.choice(characters)
        # Randomized relation/belief dimensions make format-following and counting non-trivial.
        rel_len = rng.randint(4, 12)
        bel_len = rng.randint(4, 14)
        rel_ids = random_ids("C", rel_len, rng)
        rel_block = "\n".join(
            f"{cid}: Background for {cid}; stance={rng.choice(['pro', 'neutral', 'against'])}; year={rng.randint(2018, 2026)}."
            for cid in rel_ids
        )
        belief_names = [f"Idea {i + 1}: policy-{rng.randint(100, 999)}" for i in range(bel_len)]
        belief_block = "\n".join(f"{name} : {rng.randint(0, 100)}" for name in belief_names)
        action_hist = synthetic_action_history(lines=rng.randint(8, 28), rng=rng)

        # The prompt includes a demo "case"; the original code fills it with random +/- values.
        demo_vals = [rng.randint(-max_change, max_change) for _ in range(100)]
        demo_vals_str = [f"+{v}" if v > 0 else str(v) for v in demo_vals]
        case_rel = ", ".join(demo_vals_str[: len(rel_ids)])
        case_bel = ", ".join(demo_vals_str[: bel_len])

        prompt = render_prompt(
            "prompt_4_reflect.txt",
            [
                self_char.id_name,
                self_char.self_description(),
                belief_block,
                action_hist or "No action history.",
                rel_block or "No other roles.",
                str(rel_len),
                str(bel_len),
                str(max_change),
                str(max_change),
                case_rel,
                case_bel,
            ],
        )
        resp = client.chat_completion(model=model_name, prompt=prompt, temperature=0.0, max_tokens=300)
        rel_delta, bel_delta = parse_update_output(resp)
        update_total += 1

        if rel_delta is not None and len(rel_delta) == len(rel_ids):
            nor_ok += 1
            if all(-max_change <= v <= max_change for v in rel_delta):
                rus_ok += 1
        if bel_delta is not None and len(bel_delta) == bel_len:
            nob_ok += 1
            if all(-max_change <= v <= max_change for v in bel_delta):
                bus_ok += 1

    # ---------------------------
    # NoA: # of Action History
    # ---------------------------
    noa_ok = 0
    noa_total = 0
    for _ in range(n_noa):
        n_lines = rng.randint(35, 160)
        action_hist = synthetic_action_history(lines=n_lines, rng=rng)
        gt_noa = len([ln for ln in action_hist.split("\n") if ln.strip()])
        self_char = rng.choice(characters)
        tgt_char = rng.choice([c for c in characters if c.id_name != self_char.id_name] or characters)
        prompt = render_prompt(
            "prompt_4_facechat.txt",
            [
                self_char.id_name,
                tgt_char.id_name,
                self_char.self_description(),
                tgt_char.short_description(),
                "Environment summary: you are negotiating.",
                action_hist,
                "",
                "Round 1: say hello.",
            ],
        )
        resp = client.chat_completion(model=model_name, prompt=prompt, temperature=0.0, max_tokens=200)
        got = parse_count_flexible(
            resp,
            labels=(
                r"Number of Action History",
                r"Number of Action Histories",
                r"Number of Historical Events",
                r"Action History Number",
            ),
        )
        noa_total += 1
        if got == gt_noa:
            noa_ok += 1

    # ---------------------------
    # NoCR: # of Chat Rounds
    # ---------------------------
    nocr_ok = 0
    nocr_total = 0
    for _ in range(n_nocr):
        rounds = rng.randint(8, 36)
        chat_hist = synthetic_chat_history(rounds, rng)
        self_char = rng.choice(characters)
        prompt = render_prompt(
            "prompt_wo_thinking/prompt_4_summarize_wo_thinking.txt",
            [
                self_char.id_name,
                self_char.self_description(),
                "Environment summary: keep it short.",
                chat_hist,
            ],
        )
        resp = client.chat_completion(model=model_name, prompt=prompt, temperature=0.0, max_tokens=200)
        got = parse_count_flexible(
            resp,
            labels=(
                r"Number of Chat Round",
                r"Number of Chat Rounds",
                r"Chat Round Number",
                r"Number of Action History",
            ),
        )
        nocr_total += 1
        if got == rounds:
            nocr_ok += 1

    # ---------------------------
    # NoC / NoR(resource): counts
    # ---------------------------
    def count_prompt(kind: str, label: str, ids: Sequence[str]) -> str:
        joined = "\n".join(ids)
        return (
            f"You are given a list of {kind} IDs, one per line:\n"
            f"{joined}\n\n"
            f"Output strictly in this format:\n"
            f"### {label}: <Arabic integer>\n"
            f"Do not output anything else."
        )

    noc_ok = 0
    for _ in range(n_noc):
        k = rng.randint(8, 40)
        ids = random_ids("C", k, rng)
        prompt = count_prompt("character", "Number of Character", ids)
        resp = client.chat_completion(model=model_name, prompt=prompt, temperature=0.0, max_tokens=20)
        got = parse_count_flexible(
            resp,
            labels=(
                r"Number of Character",
                r"Number of Characters",
                r"Character Number",
                r"Character Count",
            ),
        )
        if got == k:
            noc_ok += 1

    nor_res_ok = 0
    for _ in range(n_nor_res):
        k = rng.randint(8, 40)
        ids = random_ids("R", k, rng)
        prompt = count_prompt("resource", "Number of Resource", ids)
        resp = client.chat_completion(model=model_name, prompt=prompt, temperature=0.0, max_tokens=20)
        got = parse_count_flexible(
            resp,
            labels=(
                r"Number of Resource",
                r"Number of Resources",
                r"Resource Number",
                r"Resource Count",
            ),
        )
        if got == k:
            nor_res_ok += 1

    return Table3LikeRow(
        model_name=model_name,
        cs=pct(cs_ok, cs_total),
        bus=pct(bus_ok, update_total),
        rus=pct(rus_ok, update_total),
        nor_rel_len=pct(nor_ok, update_total),
        nob_bel_len=pct(nob_ok, update_total),
        noa_action_count=pct(noa_ok, noa_total),
        nocr_chat_round=pct(nocr_ok, nocr_total),
        noc_character_count=pct(noc_ok, n_noc),
        nor_resource_count=pct(nor_res_ok, n_nor_res),
    )


def print_table(rows: Sequence[Table3LikeRow]) -> None:
    headers = [
        "Model Name",
        "CS",
        "BUS",
        "RUS",
        "NoR",
        "NoB",
        "NoA",
        "NoCR",
        "NoC",
        "NoR(res)",
    ]
    data = [
        [
            r.model_name,
            fmt_pct(r.cs),
            fmt_pct(r.bus),
            fmt_pct(r.rus),
            fmt_pct(r.nor_rel_len),
            fmt_pct(r.nob_bel_len),
            fmt_pct(r.noa_action_count),
            fmt_pct(r.nocr_chat_round),
            fmt_pct(r.noc_character_count),
            fmt_pct(r.nor_resource_count),
        ]
        for r in rows
    ]
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in data)) for i in range(len(headers))]
    fmt_row = lambda row: "  ".join(row[i].ljust(col_widths[i]) for i in range(len(headers)))
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in col_widths]))
    for row in data:
        print(fmt_row(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", type=str, default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-cs", type=int, default=30)
    parser.add_argument("--n-update", type=int, default=20)
    parser.add_argument("--n-noa", type=int, default=30)
    parser.add_argument("--n-nocr", type=int, default=30)
    parser.add_argument("--n-noc", type=int, default=20)
    parser.add_argument("--n-nor-res", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1, help="Repeat benchmark with shifted seeds and average.")
    parser.add_argument("--seed-step", type=int, default=1000, help="Seed increment between repeats.")
    args = parser.parse_args()

    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    base_url = OPENAI_BASE_URL or os.environ.get("OPENAI_BASE_URL", "")
    if not api_key:
        raise SystemExit("Missing OpenAI API key. Set OPENAI_API_KEY at top of this file (or env var OPENAI_API_KEY).")
    if not base_url:
        raise SystemExit("Missing OpenAI base URL. Set OPENAI_BASE_URL at top of this file (or env var OPENAI_BASE_URL).")

    scenario_dir = Path(args.scenario_dir)
    client = OpenAIChatClient(api_key=api_key, base_url=base_url)

    rows: list[Table3LikeRow] = []
    repeats = max(1, int(args.repeats))
    for model in (OPENAI_GPT52_MODEL, OPENAI_GPT4_TURBO_MODEL, OPENAI_GPT35_MODEL):
        model_rows: list[Table3LikeRow] = []
        for rep in range(repeats):
            model_rows.append(
                benchmark_table3_like(
                    client=client,
                    model_name=model,
                    scenario_dir=scenario_dir,
                    n_cs=args.n_cs,
                    n_update=args.n_update,
                    n_noa=args.n_noa,
                    n_nocr=args.n_nocr,
                    n_noc=args.n_noc,
                    n_nor_res=args.n_nor_res,
                    seed=args.seed + rep * int(args.seed_step),
                )
            )
        rows.append(average_rows(model, model_rows))

    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

