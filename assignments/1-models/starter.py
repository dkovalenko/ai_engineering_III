# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "litellm",
# ]
# ///
"""
HW1 Starter — Foundational Models & Sampling Parameters

Запускай цей скрипт окремо для кожного кейсу (1–4):
  1. Встанови CASE_NAME (рядок нижче) відповідно до кейсу.
  2. Введи свій PROMPT для цього кейсу.
  3. Розкоментуй потрібні моделі в SMALL_MODELS та LARGE_MODELS.
  4. uv run starter.py

Результати зберігаються в responses/<CASE_NAME>/.
Кейс 5 (Capability Ladder) — окремий скрипт: uv run ladder.py
Загалом: 4 кейси × 6 запусків = 24 автоматичні запуски.

──────────────────────────────────────────────
Setup — Large model (хмара, через OpenRouter):
  export OPENROUTER_API_KEY="sk-or-..."

Setup — Small model (локально, через Ollama):
  1. Встановити: https://ollama.com
  2. Завантажити модель: ollama pull gemma3:2b
  3. Розкоментувати рядок нижче в SMALL_MODELS

Setup — Small model (хмара, через OpenRouter):
  Розкоментувати "openrouter/google/gemma-3-4b-it" в SMALL_MODELS

Top models (вибери одну):
  - anthropic/claude-sonnet-4-6  — Claude Sonnet 4.6 (основна)
  - z-ai/glm-5.2                 — GLM-5.2 від Zhipu AI (альтернатива)
──────────────────────────────────────────────
"""

import time
from pathlib import Path

import litellm
from litellm import completion

litellm.suppress_debug_info = True

# ── Змінюй цей рядок для кожного кейсу ──────────────────────────────────────
CASE_NAME = "case1_data_extraction"
# Варіанти:
#   "case1_data_extraction"
#   "case2_summarization"
#   "case3_reasoning"
#   "case4_creative_pitch"
# (Кейс 5 = Capability Ladder — запускай ladder.py)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = ""  # необов'язково — інструкція для моделі (що робити, формат відповіді)
PROMPT = "YOUR PROMPT HERE"  # контекст або запитання (текст логу, правила, тощо)

# ── Large models (хмара) — розкоментуй один ──────────────────────────────────
LARGE_MODELS = [
    "openrouter/anthropic/claude-sonnet-4-6",
    # "openrouter/z-ai/glm-5.2",
    # "openrouter/meta-llama/llama-4-scout",
]

# ── Small / Local models — розкоментуй один ──────────────────────────────────
SMALL_MODELS = [
    # Варіант A: платна мала модель через OpenRouter (~$0.04/M токенів, найпростіше)
    # "openrouter/google/gemma-3-4b-it",

    # Варіант B: локально через Ollama (безкоштовно, потребує ~2 ГБ RAM)
    # "ollama/gemma3:2b",

    # Варіант C: хмара безкоштовно через OpenRouter (є rate limit, може не пройти)
    # "openrouter/meta-llama/llama-3.2-3b-instruct:free",
]

MODELS = LARGE_MODELS + SMALL_MODELS

TEMPERATURES = [0.1, 0.7, 1.2]
TOP_P = 0.9
SEED = 42

# Стеля довжини відповіді, не параметр семплювання — Temperature і Top-P лишаються як були.
# На OpenRouter max_tokens — спільний бюджет: приховані reasoning/thinking токени
# плюс видимий текст. 1024 для Gemini, o-series, Claude thinking, DeepSeek R1 часто
# закінчується ще на міркуванні, і текст приходить обрізаним (finish_reason="length").
# 8192 лишає запас на короткі відповіді цього ДЗ. Це стеля, не ціль: модель
# зупиняється, коли закінчила, і не дописує відповідь до ліміту.
MAX_TOKENS = 8192

# Необов'язково. None = рівень thinking за замовчуванням моделі (нічого не перезаписуємо).
# Можна "low" / "medium" / "high". Не кожна модель приймає "none"; у Gemini 3.8 немає "minimal".
# Це окремий важіль від Temperature / Top-P — не крути його, щоб «полагодити» семплювання.
REASONING_EFFORT = None


def _field(obj, key):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def token_counts(resp):
    """(completion_tokens, reasoning_tokens). Будь-яке значення може бути None."""
    usage = _field(resp, "usage")
    completion = _field(usage, "completion_tokens")
    details = _field(usage, "completion_tokens_details")
    reasoning = _field(details, "reasoning_tokens")
    return completion, reasoning


def usage_line(finish_reason, completion_tokens, reasoning_tokens) -> str:
    parts = []
    if finish_reason:
        parts.append(f"finish: {finish_reason}")
    if completion_tokens is not None:
        parts.append(f"completion_tokens: {completion_tokens}")
    if reasoning_tokens is not None:
        parts.append(f"reasoning_tokens: {reasoning_tokens}")
        if completion_tokens is not None and reasoning_tokens <= completion_tokens:
            parts.append(f"visible_tokens: {completion_tokens - reasoning_tokens}")
    return " | ".join(parts)


def truncation_warning(finish_reason, text, completion_tokens, reasoning_tokens):
    """Пояснення, якщо відповідь обрізана. None, коли все гаразд."""
    hit_limit = finish_reason in ("length", "max_tokens")
    thinking_ate_budget = not (text or "").strip() and (reasoning_tokens or 0) > 0
    if not hit_limit and not thinking_ate_budget:
        return None

    lines = ["УВАГА: видима відповідь обрізана або порожня."]
    if hit_limit:
        lines.append(
            f"finish_reason={finish_reason!r}: модель зупинилась, "
            f"бо вперлася в max_tokens={MAX_TOKENS}."
        )
    if reasoning_tokens:
        detail = f"reasoning_tokens={reasoning_tokens}"
        if completion_tokens is not None:
            detail += f", completion_tokens={completion_tokens}"
            if reasoning_tokens <= completion_tokens:
                detail += f", visible_tokens={completion_tokens - reasoning_tokens}"
        lines.append(detail + ".")
        lines.append(
            "На OpenRouter токени міркування (reasoning) входять у той самий max_tokens, "
            "що й видимий текст. Підніми MAX_TOKENS на початку цього файлу."
        )
    else:
        lines.append(
            "Підніми MAX_TOKENS на початку цього файлу. Для reasoning-моделей "
            "(Gemini, o-series, Claude thinking, DeepSeek R1) thinking-токени "
            "теж витрачають цей бюджет."
        )
    return "\n".join(lines)


def ask(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = TOP_P,
    seed: int | None = SEED,
    system: str = "",
):
    """Returns a dict: text, elapsed_s, cost, finish_reason, token counts, warning."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    request = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_tokens": MAX_TOKENS,
    }
    if REASONING_EFFORT:
        request["reasoning_effort"] = REASONING_EFFORT

    start = time.time()
    resp = completion(**request)
    elapsed = time.time() - start
    cost = (getattr(resp, "_hidden_params", None) or {}).get("response_cost", None)

    choice = resp.choices[0]
    raw = choice.message.content
    text = raw if isinstance(raw, str) else ""
    finish_reason = _field(choice, "finish_reason")
    completion_tokens, reasoning_tokens = token_counts(resp)
    warning = truncation_warning(finish_reason, text, completion_tokens, reasoning_tokens)
    return {
        "text": text,
        "elapsed": elapsed,
        "cost": cost,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "warning": warning,
    }


def save_response(
    out_dir: Path,
    short_name: str,
    temp: float,
    top_p: float,
    text: str,
    elapsed: float,
    cost,
    usage: str = "",
    warning: str | None = None,
):
    case_dir = out_dir / CASE_NAME
    case_dir.mkdir(parents=True, exist_ok=True)
    slug = short_name.replace("/", "_").replace(" ", "_").replace(":", "_")
    path = case_dir / f"{slug}_t{temp}_p{top_p}.txt"
    cost_str = f"${cost:.6f}" if cost is not None else "n/a"
    header = f"# temperature: {temp} | top_p: {top_p} | time: {elapsed:.1f}s | cost: {cost_str}"
    if usage:
        header += f" | {usage}"
    body = text or ""
    if warning:
        body = warning + "\n\n" + body
    path.write_text(header + "\n" + body + "\n")
    print(f"  → saved {path}")


def main():
    out_dir = Path(__file__).parent / "responses"

    if not MODELS:
        print("ERROR: MODELS is empty. Розкоментуй хоча б одну модель у SMALL_MODELS або LARGE_MODELS.")
        return

    print("=" * 70)
    print(f"Case:   {CASE_NAME}")
    print(f"Prompt: {PROMPT[:80]}{'…' if len(PROMPT) > 80 else ''}")
    print("=" * 70)

    for model in MODELS:
        for temp in TEMPERATURES:
            short_name = model.removeprefix("openrouter/").removeprefix("ollama/")
            tier = "LARGE" if model in LARGE_MODELS else "SMALL"
            print(f"\n{'─' * 70}")
            print(f"[{tier}] Model: {short_name}  |  T={temp}  |  top_p={TOP_P}")
            print("─" * 70)
            try:
                result = ask(model, PROMPT, temp, system=SYSTEM_PROMPT)
                cost = result["cost"]
                cost_str = f"${cost:.6f}" if cost is not None else "n/a"
                usage = usage_line(
                    result["finish_reason"],
                    result["completion_tokens"],
                    result["reasoning_tokens"],
                )
                print(f"Time: {result['elapsed']:.1f}s | Cost: {cost_str}")
                if usage:
                    print(usage)
                if result["warning"]:
                    print(result["warning"])
                print(result["text"])
                save_response(
                    out_dir,
                    short_name,
                    temp,
                    TOP_P,
                    result["text"],
                    result["elapsed"],
                    cost,
                    usage,
                    result["warning"],
                )
            except Exception as e:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
