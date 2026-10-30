"""AI によるタスク計画: 自然言語の指示から、ピラミッドの 10 か所それぞれに置くブロックの色を決める。

方式（どれも同じ PyramidPlan のスキーマで返す）:
  ollama - ローカル LLM（既定は gemma4:e4b）。Ollama の構造化出力を使う。API キー不要でオフラインで動く
  claude - Claude API の構造化出力（ANTHROPIC_API_KEY などの認証情報が必要）
  rules  - LLM が使えない・個数と矛盾する計画を返した場合に使う、決定的な代替手段

LLM の計画が個数と矛盾していたら、何が足りないかを伝えて 1 回だけ作り直させる（自己修正）。
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from .env import COLORS, PYRAMID_ROWS

CLAUDE_MODEL = "claude-opus-5-5"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
# Gemma 4 の思考モード（答える前に推論を書き出す。Ollama の既定でも有効）。0 にすると速いが計画の正しさが下がる
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "1") == "1"
# 1 = 段の数え方・上下の向きの規則を加えたプロンプト（v2）、0 = 改善前のプロンプト（比較実験用）
PROMPT_V2 = os.environ.get("PLANNER_PROMPT_V2", "1") == "1"

Color = Literal["red", "orange", "green", "blue", "purple", "lime", "pink", "cyan", "brown"]


class PyramidPlan(BaseModel):
    rows: list[list[Color]] = Field(
        description="Colour of every block. rows[0] is the bottom row (4 blocks), then 3, 2 and 1 (top). "
        "Within a row, blocks are listed from the robot side to the far side."
    )
    explanation: str = Field(description="One short sentence, in the instruction's language, explaining the plan.")


SYSTEM = (
    "You plan a block-stacking task for an SO-101 robot arm. It builds a pyramid of rows "
    f"{', '.join(map(str, PYRAMID_ROWS))} blocks from the bottom up. Choose the colour of every block following "
    "the user's instruction. Colours: red, orange, green, blue, purple, lime (yellow-green), pink, cyan (light blue), "
    "brown. Warm colours are red, orange, pink and brown; cool colours are blue, cyan, green and purple. "
    "In each row, the first and last blocks are the outer ones and the others are inner. "
    "You may use each colour at most as many times as it was detected. "
    "If the instruction is impossible or ambiguous, choose the closest feasible plan and say why in the explanation. "
    "Write the explanation as ONE short sentence (at most 60 characters) in the same language as the instruction; "
    "for a Japanese instruction, answer in Japanese."
)
# v2: scripts/eval_planner.py の開発用ケースで見つかった失敗（上から数える指示の上下の取り違え、段ごとの色を
# 1 個ずつ循環させてしまう）を防ぐための規則
SYSTEM_V2_RULES = (
    " Rules: rows[0] is the BOTTOM row (the base, 土台, 1段目) and rows[3] is the TOP (頂上, 一番上, 4段目). "
    "If the instruction lists colours from the top (e.g. 'from top to bottom', '一番上は…その下は…', '上から'), "
    "map the first colour to rows[3] and continue downwards. "
    "When a colour is given for a row (or for 'the rest'), EVERY block in that row gets that colour; never cycle "
    "the listed colours block by block unless the instruction asks for mixing or alternating within a row. "
    "Before answering, check every row against the instruction."
)
if PROMPT_V2:
    SYSTEM += SYSTEM_V2_RULES


def _user_prompt(instruction: str, counts: Counter) -> str:
    detected = ", ".join(f"{c} x{n}" for c, n in counts.most_common())
    return f"Detected blocks: {detected}\nInstruction: {instruction}"


def problems(plan: PyramidPlan, counts: Counter) -> list[str]:
    """計画の問題点（空なら実行できる）。LLM に作り直させるときにそのまま伝える。"""
    issues = []
    if [len(r) for r in plan.rows] != list(PYRAMID_ROWS):
        issues.append(
            f"the rows must have {', '.join(map(str, PYRAMID_ROWS))} blocks, "
            f"but you gave {', '.join(str(len(r)) for r in plan.rows)}"
        )
    used = Counter(c for r in plan.rows for c in r)
    for c, n in used.items():
        if n > counts[c]:
            issues.append(f"{c} is used {n} times but only {counts[c]} were detected")
    return issues


def is_feasible(plan: PyramidPlan, counts: Counter) -> bool:
    return not problems(plan, counts)


def _ollama_chat(messages: list[dict], model: str) -> str:
    import ollama

    response = ollama.chat(
        model=model,
        messages=messages,
        format=PyramidPlan.model_json_schema(),
        options={"temperature": 0},
        think=OLLAMA_THINK,
    )
    return response.message.content


def plan_with_ollama(instruction: str, counts: Counter, model: str = OLLAMA_MODEL) -> PyramidPlan:
    """Ollama 経由のローカル LLM。`format` で出力を PyramidPlan の JSON スキーマに制約する。
    個数と矛盾していたら、問題点を伝えて 1 回だけ作り直させる。"""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _user_prompt(instruction, counts)}]
    content = _ollama_chat(messages, model)
    plan = PyramidPlan.model_validate_json(content)
    issues = problems(plan, counts)
    if issues:
        print(f"[planner] asking {model} to fix: {'; '.join(issues)}")
        messages += [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "That plan is not feasible: "
                + "; ".join(issues)
                + ". Please give a corrected plan that follows the instruction as closely as possible.",
            },
        ]
        plan = PyramidPlan.model_validate_json(_ollama_chat(messages, model))
    return plan


def plan_with_claude(instruction: str, counts: Counter) -> PyramidPlan:
    import anthropic

    client = anthropic.Anthropic()
    response = client.beta.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": _user_prompt(instruction, counts)}],
        output_format=PyramidPlan,
        output_config={"effort": "low"},
        # 安全分類器に拒否された場合、同じ呼び出しの中で代替モデルで再試行される
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"Claude returned no plan (stop_reason={response.stop_reason})")
    return response.parsed_output


_COLOR_WORDS = {
    "red": ["red", "赤"],
    "orange": ["orange", "オレンジ", "橙"],
    "green": ["green", "緑"],
    "blue": ["blue", "青"],
    "purple": ["purple", "紫"],
    "lime": ["lime", "yellow-green", "黄緑"],
    "pink": ["pink", "ピンク"],
    "cyan": ["cyan", "light blue", "水色"],
    "brown": ["brown", "茶"],
}


def plan_with_rules(instruction: str, counts: Counter) -> PyramidPlan:
    """オフラインの代替手段: 指示に出てくる色を、出てくる順に下の段から 1 段ずつ割り当て、残りの段は最後の色に
    する（「交互」の指示や、段の数より多くの色が出てくるときは、1 個ずつ順に割り当てる）。足りなくなった色は飛ばし、
    色の言及がない・足りないときは、残りの多い色で埋める。"""
    text = instruction.lower()

    # 長い色名を先に照合し、light blue 内の blue や黄緑内の緑を二重に拾わない。
    words = {word: color for color, names in _COLOR_WORDS.items() for word in names}
    patterns = [
        rf"(?<![a-z]){re.escape(word)}(?![a-z])" if word.isascii() else re.escape(word)
        for word in sorted(words, key=len, reverse=True)
    ]
    mentioned = list(dict.fromkeys(words[m.group()] for m in re.finditer("|".join(patterns), text)))
    if any(k in text for k in ["上から", "一番上は", "top first", "from the top", "top to bottom"]):
        mentioned.reverse()
    per_block = len(mentioned) > len(PYRAMID_ROWS) or any(k in text for k in ["交互", "alternat"])
    left = Counter(counts)
    rows: list[list[str]] = []
    k = 0
    for row_i, n in enumerate(PYRAMID_ROWS):
        row = []
        for _ in range(n):
            if per_block:
                pool = [c for c in mentioned if left[c] > 0]
                c = pool[k % len(pool)] if pool else max(left, key=left.get)
                k += 1
            else:
                # この段の色（言及が段の数より少なければ最後の色）。足りなければ後ろの色、それもなければ多い色
                want = mentioned[min(row_i, len(mentioned) - 1) :] if mentioned else []
                c = next((c for c in want if left[c] > 0), None) or max(left, key=left.get)
            if left[c] <= 0:
                raise ValueError(f"not enough blocks for a {'-'.join(map(str, PYRAMID_ROWS))} pyramid: {dict(counts)}")
            left[c] -= 1
            row.append(c)
        rows.append(row)
    how = "指示に出てくる色を順に" if mentioned else "数の多い色から順に"
    return PyramidPlan(rows=rows, explanation=f"{how}下の段から割り当てました。")


def plan_unlimited(instruction: str, backend: str = "ollama") -> tuple[PyramidPlan, str]:
    """ブロックの数に制限がないとして指示を計画する（テーブルに置くブロックを決めるため）。

    この計画は、検出したブロックの個数と矛盾しなければ本番でもそのまま使う（main.py）。LLM に 2 回聞くと、
    温度 0 でも 2 回目に違う色を選ぶことがあり、テーブルに余分のない色を使って失敗したため。"""
    unlimited = Counter({c: sum(PYRAMID_ROWS) for c in COLORS})
    return make_plan(instruction, list(unlimited.elements()), backend)


def required_colors(plan: PyramidPlan) -> Counter:
    """計画に必要な色ごとの個数。"""
    return Counter(c for row in plan.rows for c in row)


BACKENDS = {"ollama": plan_with_ollama, "claude": plan_with_claude}
PLANNER_LABELS = {"ollama": f"ローカルLLM {OLLAMA_MODEL}", "claude": f"Claude {CLAUDE_MODEL}", "rules": "ルールベース"}


def make_plan(instruction: str, detected_colors: list[str], backend: str = "ollama") -> tuple[PyramidPlan, str]:
    """(計画, 使った方式) を返す。backend: 'ollama' | 'claude' | 'rules' | 'auto'（ollama → claude の順）。"""
    counts = Counter(detected_colors)
    order = {"auto": ["ollama", "claude"], "rules": []}.get(backend, [backend])
    for name in order:
        try:
            plan = BACKENDS[name](instruction, counts)
            issues = problems(plan, counts)
            if not issues:
                return plan, name
            print(f"[planner] {name} returned an infeasible plan {plan.rows}: {'; '.join(issues)}")
        # どんな失敗（サーバー未起動、モデル未取得、認証情報なし、拒否、壊れた JSON など）でもルールベースへ切り替える
        except Exception as e:  # noqa: BLE001
            print(f"[planner] {name} unavailable ({type(e).__name__}: {e})")
    if order:
        print("[planner] falling back to the rule-based planner")
    return plan_with_rules(instruction, counts), "rules"


assert set(COLORS) == set(Color.__args__), "keep planner colours in sync with env.COLORS"
