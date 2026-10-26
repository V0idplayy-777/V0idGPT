#!/usr/bin/env python3
"""Last-mile curriculum: recover from your own wrong answer, and borrow subtraction.

The main corpus (gen_dataset.py) teaches the model to *stand behind* a correct answer when the
user pushes back — that's deliberate, it stops the assistant from agreeing with "2 + 2 = 5". The
gap it leaves is the opposite case: when the assistant's earlier answer was actually wrong and the
user re-asks, it should recompute instead of repeating itself.

This generator writes three kinds of conversation, all answers computed and re-checked here:

  recompute   a wrong answer followed by the same question again -> correct answer + admission
  borrow      two-digit subtraction requiring a borrow, with a written decomposition
  chat        short non-numeric turns, so the pass cannot collapse into number-answering

Usage:
  python3 scripts/gen_fixups.py --out /tmp/vg/fixups.jsonl --recompute 250 --borrow 250
"""
import argparse
import json
import random
import re

RNG = random.Random(4242)

SYSTEMS = [
    "You are v0idgpt, a small assistant running locally in the browser. Answer what you're asked, be concise, and admit uncertainty.",
    "You are v0idgpt, a compact on-device assistant. Prefer short, direct answers.",
    "You are v0idgpt. Be brief, friendly and accurate. If you're unsure, say so.",
]
ADD_Q = ["what's {a} + {b}", "what is {a} plus {b}", "{a} + {b}", "compute {a} + {b}", "{a} plus {b}?"]
SUB_Q = ["what's {a} - {b}", "what is {a} minus {b}", "{a} - {b}", "subtract {b} from {a}", "compute {a} - {b}"]
ASK_AGAIN = ["no, I said {q}", "that's not what I asked — {q}", "I said {q}", "no. {q}", "wrong, {q} again"]


def conv(*pairs):
    msgs = []
    for role, content in pairs:
        msgs.append({"role": role, "content": content})
    return {"system": RNG.choice(SYSTEMS), "messages": msgs}


def wrong_answer(r):
    """A plausible neighbour of the right answer: never equal to it."""
    for _ in range(20):
        d = RNG.choice([-9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        w = r + d
        if w != r and w > 0:
            return w
    return r + 2


def gen_recompute(n):
    out = []
    while len(out) < n:
        op = RNG.choice(["+", "-", "×"])
        if op == "+":
            a, b = RNG.randint(2, 99), RNG.randint(2, 99)
            r = a + b
            q = RNG.choice(ADD_Q).format(a=a, b=b)
            fix = f"Right — {a} + {b} = {r}. I had that wrong. My mistake."
            maths = f"{a} + {b} = {r}"
        elif op == "-":
            a, b = RNG.randint(30, 99), RNG.randint(10, 29)
            if b >= a:
                continue
            r = a - b
            q = RNG.choice(SUB_Q).format(a=a, b=b)
            fix = f"Let me redo it: {a} - {b} = {r}. I had that wrong. My mistake."
            maths = f"{a} - {b} = {r}"
        else:
            a, b = RNG.randint(3, 12), RNG.randint(3, 12)
            r = a * b
            q = RNG.choice(["what's {a} times {b}", "what is {a} x {b}", "{a} * {b}"]).format(a=a, b=b)
            fix = f"Let me redo it: {a} × {b} = {r}. I had that wrong. My mistake."
            maths = f"{a} × {b} = {r}"
        if RNG.random() < 0.5:
            first = f"{wrong_answer(r)}."
        else:
            first = f"I think it's {wrong_answer(r)}."
        again = RNG.choice(ASK_AGAIN).format(q=q)
        out.append(conv(("user", q), ("assistant", first), ("user", again), ("assistant", fix)))
    return out


def gen_borrow(n):
    out = []
    while len(out) < n:
        a, b = RNG.randint(31, 99), RNG.randint(12, 88)
        if b >= a or (a % 10) >= (b % 10):
            continue
        r = a - b
        q = RNG.choice(SUB_Q).format(a=a, b=b)
        t = b // 10 * 10
        rest = b % 10
        ans = f"{r}. {a} - {t} = {a - t}, then {a - t} - {rest} = {r}."
        out.append(conv(("user", q), ("assistant", ans)))
    return out


CHAT = [
    ("hi", ["Hey! What are you up to?", "Hi — how's your day going?", "Hello! What's on your mind?"]),
    ("how are you", ["Doing well, thanks. What about you?",
                     "Pretty good — I'm just a small model in a browser tab, but I'm happy to be here.",
                     "All good. How's your day been?"]),
    ("what can you do", ["I can talk things through, explain ideas, help with short code, and do arithmetic. What do you need?",
                         "Short answers, explanations, and small maths. Ask away."]),
    ("what's your name", ["I'm v0idgpt.", "v0idgpt — a small model running locally in your browser."]),
    ("thanks", ["Anytime.", "No problem.", "Happy to help."]),
    ("what are you", ["A small language model running entirely in your browser — no server, no API.",
                      "I'm a 135M-parameter transformer compiled to run on your machine."]),
    ("good morning", ["Morning! What's the plan today?", "Good morning — how'd you sleep?"]),
    ("tell me something interesting", ["Octopuses have three hearts and blue blood. Two pump to the gills, one to the body.",
                                       "Honey never really goes off — sealed jars from thousands of years ago are still edible."]),
    ("i had a rough day", ["That's rough. Want to talk it through or take your mind off it?",
                           "Sorry to hear it. What happened?"]),
    ("i'm learning to play guitar", ["Nice. Start with the C, G, and D chords — they unlock a lot of songs.",
                                     "Guitar is a great one. Chords first, then strumming patterns."]),
]


def gen_chat(rounds=4):
    out = []
    for _ in range(rounds):
        for q, answers in CHAT:
            out.append(conv(("user", q), ("assistant", RNG.choice(answers))))
    return out


ARITH_RX = [
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), lambda a, b: a + b),
    (re.compile(r"(\d+)\s*-\s*(\d+)"), lambda a, b: a - b),
    (re.compile(r"(\d+)\s*(?:×|x|\*)\s*(\d+)"), lambda a, b: a * b),
]


def validate(convs):
    bad = 0
    for c in convs:
        msgs = c["messages"]
        q = msgs[0]["content"]
        for rx, fn in ARITH_RX:
            m = rx.search(q)
            if not m:
                continue
            expect = fn(int(m.group(1)), int(m.group(2)))
            # the final assistant turn must contain the true answer
            final = msgs[-1]["content"]
            if expect not in {int(x) for x in re.findall(r"\d+", final)}:
                bad += 1
                print("MISMATCH", q, "->", final, "expected", expect)
            break
    if bad:
        raise SystemExit(f"{bad} mismatched final answers")
    return len(convs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/vg/fixups.jsonl")
    ap.add_argument("--recompute", type=int, default=250)
    ap.add_argument("--borrow", type=int, default=250)
    ap.add_argument("--chat-rounds", type=int, default=4)
    args = ap.parse_args()

    convs = gen_recompute(args.recompute) + gen_borrow(args.borrow) + gen_chat(args.chat_rounds)
    validate(convs)
    RNG.shuffle(convs)
    with open(args.out, "w") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(convs)} conversations -> {args.out}")


if __name__ == "__main__":
    main()
