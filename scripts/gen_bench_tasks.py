#!/usr/bin/env python3
"""Sprint 4.1: generate the 3-task agentic benchmark suite.

Emits benchmarks/tasks/{toolcall,multiturn,summarize}.jsonl — 10 prompts each.
Deterministic (no RNG) so raw dumps are reproducible. Summarize prompts are
~4-8k ctx by repeating structured report sections.

Each line: {"id","task","kind","messages":[...],"max_tokens",
           "tool_choice"?,"tools"?}. `messages` is chat-format; the summarize
task uses single-turn long prompts.
"""
import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "tasks")
os.makedirs(OUT, exist_ok=True)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string", "enum": ["metric", "imperial"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search flights between two airports on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "date": {"type": "string", "format": "date"},
                },
                "required": ["from", "to", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a read-only SQL SELECT against the analytics DB.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    },
]

TOOLCALL = [
    "What's the weather in Reykjavik in metric?",
    "Book context: find flights from SFO to NRT on 2026-09-14.",
    "How hot is it in Phoenix right now, in fahrenheit?",
    "Get me the count of active users from the users table.",
    "Weather for Cape Town, metric units.",
    "Find flights from BER to JFK tomorrow (2026-08-30).",
    "SELECT total revenue from orders for last month — run it.",
    "Weather in Buenos Aires, then tell me if I need a jacket.",
    "Which airport code is for London Heathrow? Search flights LHR to LAX 2026-10-01.",
    "Run a query for the top 10 products by units sold.",
]

# multiturn: 3-5 round tool loops. Round-trip a synthetic tool result each turn.
TOOL_RESULT = {
    "get_weather": '{"temp_c":18,"condition":"cloudy","wind_kph":22}',
    "search_flights": '{"results":[{"carrier":"JL","price":780,"stops":0},{"carrier":"NH","price":690,"stops":1}]}',
    "run_query": '{"rows":[{"count":4213}],"elapsed_ms":88}',
}


def multiturn(idx):
    city = ["Oslo", "Lima", "Seoul", "Cairo", "Vienna", "Hanoi", "Quito",
            "Riga", "Tallinn", "Bogota"][idx % 10]
    msgs = [{"role": "user", "content": f"Weather in {city}?"}]
    msgs.append({"role": "assistant", "content": f"Calling get_weather for {city}."})
    msgs.append({"role": "tool", "name": "get_weather",
                 "content": TOOL_RESULT["get_weather"]})
    msgs.append({"role": "user", "content": f"Now flights from {city} to Singapore on 2026-11-02."})
    msgs.append({"role": "assistant", "content": "Calling search_flights."})
    msgs.append({"role": "tool", "name": "search_flights",
                 "content": TOOL_RESULT["search_flights"]})
    msgs.append({"role": "user", "content": "Cheapest option and does it beat booking the query for seat prices? Run a query for average fare."})
    msgs.append({"role": "assistant", "content": "Calling run_query."})
    msgs.append({"role": "tool", "name": "run_query", "content": TOOL_RESULT["run_query"]})
    return msgs


REPORT_SECTIONS = [
    "Executive summary: quarterly performance shows throughput improved across "
    "all regions while unit economics held steady. Opex increased modestly due "
    "to infrastructure scaling ahead of demand growth.",
    "Regional breakdown: APAC led growth at 23% QoQ, EMEA held at 11%, and "
    "Americas contributed 8%. Currency headwinds in EMEA offset part of the "
    "volume gain. LatAm remains a small but fast-growing base.",
    "Product line: the platform tier drove 61% of new ARR. Legacy on-prem "
    "renewals declined 14% as migration incentives took effect. Add-on modules "
    "saw the highest expansion revenue per account.",
    "Infrastructure: compute spend grew 19% against 31% growth in served "
    "tokens, indicating improving unit cost. Storage tiering recovered an "
    "estimated 7% of egress. Reliability SLOs met except for a single SEV-2 "
    "during the mid-quarter deploy.",
    "Risk register: concentration risk in two hyperscale accounts; mitigation "
    "is pipeline diversification. Supply-chain exposure to GPU allocation is "
    "hedged via multi-quarter reservations. Regulatory: data-residency rollout "
    "on track for the EMEA target date.",
    "Outlook: guidance raised to reflect the APAC momentum, tempered by the "
    "Americas renewal cycle. Hiring plan holds flat; capital allocation "
    "prioritizes inference efficiency work this half.",
]


def summarize(idx):
    # expand to ~4-8k ctx: repeat the section corpus with per-prompt framing
    corpus = " ".join(REPORT_SECTIONS)
    reps = 4 + (idx % 4)  # 4-7 repeats -> ~5-9k tokens
    body = (corpus + " ") * reps
    return ("Summarize the following quarterly report in exactly 4 bullet "
            "points, no preamble:\n\n" + body)


def main():
    def dump(name, rows):
        path = os.path.join(OUT, name)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {path} ({len(rows)} prompts)")

    tc = []
    for i, p in enumerate(TOOLCALL):
        tc.append({"id": f"toolcall-{i:02d}", "task": "toolcall",
                   "kind": "single_turn_tool",
                   "messages": [{"role": "user", "content": p}],
                   "tools": TOOLS, "tool_choice": "required",
                   "max_tokens": 256})
    dump("toolcall.jsonl", tc)

    mt = []
    for i in range(10):
        mt.append({"id": f"multiturn-{i:02d}", "task": "multiturn",
                   "kind": "multi_turn_tool_loop",
                   "messages": multiturn(i), "tools": TOOLS,
                   "tool_choice": "none", "max_tokens": 256})
    dump("multiturn.jsonl", mt)

    sm = []
    for i in range(10):
        msgs = [{"role": "user", "content": summarize(i)}]
        sm.append({"id": f"summarize-{i:02d}", "task": "summarize",
                   "kind": "long_prompt_short_output",
                   "messages": msgs, "max_tokens": 128})
    dump("summarize.jsonl", sm)


if __name__ == "__main__":
    main()
