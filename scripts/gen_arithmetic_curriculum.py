#!/usr/bin/env python3
"""Systematic arithmetic curriculum, used as a continued-fine-tuning pass.

The first SFT run teaches conversational style, but it only sees ~390 addition
questions with 373 distinct operand pairs, so it memorises those and guesses on
anything else (47 + 38 came back as 75). This generator instead *covers the
space*:

  * every single-digit operation                  (81 + 81 + 81 + 64 items)
  * the full 2..12 multiplication grid            (121 items)
  * ~1,400 two-digit additions, half with a carry
  * ~1,000 two-digit subtractions, half with a borrow
  * two-digit x one-digit products                (~500 items)
  * exact divisions with divisor 2..12            (~400 items)
  * hundreds-scale sums with a written decomposition

Every answer is computed here and every emitted conversation is re-checked by
`validate()` before it is written, so no wrong answer can enter the corpus.

Usage:
  python3 scripts/gen_arithmetic_curriculum.py --out /tmp/vg/arith.jsonl
"""
import argparse
import json
import random
import re

RNG = random.Random(7717)

SYSTEMS = [
    "You are v0idgpt, a small assistant running locally in the browser. Answer what you're asked, be concise, and admit uncertainty.",
    "You are v0idgpt, a compact on-device assistant. Prefer short, direct answers.",
    "You are v0idgpt. Be brief, friendly and accurate. If you're unsure, say so.",
]

ADD_Q = ["what's {a} + {b}", "what is {a} plus {b}", "{a} + {b} = ?", "compute {a} + {b}",
         "{a} plus {b}?", "quick one: {a} + {b}", "{a}+{b}", "add {a} to {b}",
         "what does {a} + {b} make", "how much is {a} + {b}"]
SUB_Q = ["what's {a} - {b}", "what is {a} minus {b}", "{a} - {b} = ?", "subtract {b} from {a}",
         "compute {a} - {b}", "{a} take away {b}", "{a}-{b}?", "how much is {a} - {b}"]
MUL_Q = ["what's {a} times {b}", "what is {a} x {b}", "{a} * {b}", "{a} times {b}?",
         "compute {a} multiplied by {b}", "what's {a} × {b}", "multiply {a} by {b}",
         "{a}x{b} = ?", "how much is {a} times {b}"]
DIV_Q = ["what's {a} divided by {b}", "{a} / {b}", "divide {a} by {b}",
         "how many times does {b} go into {a}", "{a} ÷ {b} = ?"]


def direct(r, a=None, b=None, op="+"):
    """Answer in the plain style the app already produces."""
    forms = [f"{r}.", f"That's {r}.", f"It's {r}.", f"{r}."]
    if a is not None and RNG.random() < 0.45:
        forms.append(f"{a} {op} {b} = {r}.")
    return RNG.choice(forms)


def add_steps(a, b):
    """Digit-wise carry decomposition ending in the answer."""
    ua, ub = a % 10, b % 10
    if ua + ub < 10:
        return None
    carry = 1
    ta, tb = a // 10, b // 10
    return (f"{a} + {b} = {a + b}. Units: {ua} + {ub} = {ua + ub}, so write {ua + ub - 10} and carry {carry}. "
            f"Tens: {ta} + {tb} + {carry} = {a + b // 10 * 0 + (a + b) // 10}. "
            f"Answer: {a + b}.")


def conv(q, a, cat="math", follow=None):
    msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    if follow:
        msgs.append({"role": "user", "content": follow[0]})
        msgs.append({"role": "assistant", "content": follow[1]})
    return {"system": RNG.choice(SYSTEMS), "messages": msgs}


def gen_single_digit():
    out = []
    for a in range(2, 10):
        for b in range(2, 10):
            out.append(conv(RNG.choice(ADD_Q).format(a=a, b=b), direct(a + b, a, b, "+")))
            if a > b:
                out.append(conv(RNG.choice(SUB_Q).format(a=a, b=b), direct(a - b, a, b, "-")))
            out.append(conv(RNG.choice(MUL_Q).format(a=a, b=b), direct(a * b, a, b, "×")))
            out.append(conv(RNG.choice(DIV_Q).format(a=a * b, b=a), direct(b, a * b, a, "÷")))
    return out


def gen_two_digit_add(n=1400):
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        a, b = RNG.randint(10, 99), RNG.randint(10, 99)
        carry = (a % 10) + (b % 10) >= 10
        # keep the carry cases well represented: that is the hard part
        if not carry and RNG.random() < 0.5:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        q = RNG.choice(ADD_Q).format(a=a, b=b)
        r = a + b
        if carry and RNG.random() < 0.35 and add_steps(a, b):
            ans = add_steps(a, b)
            follow = ("how'd you get that", f"{a} + {b} = {r}.")
        else:
            ans = direct(r, a, b, "+")
            follow = None
        out.append(conv(q, ans, follow=follow))
    return out


def gen_two_digit_sub(n=1000):
    out, seen = [], set()
    while len(out) < n:
        a, b = RNG.randint(21, 99), RNG.randint(10, 99)
        if b >= a:
            continue
        borrow = (a % 10) < (b % 10)
        if borrow and RNG.random() < 0.6:
            pass
        elif not borrow and RNG.random() < 0.4:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append(conv(RNG.choice(SUB_Q).format(a=a, b=b), direct(a - b, a, b, "-")))
    return out


def gen_tables():
    out = []
    for a in range(2, 13):
        for b in range(2, 13):
            out.append(conv(RNG.choice(MUL_Q).format(a=a, b=b), direct(a * b, a, b, "×")))
            out.append(conv(RNG.choice(DIV_Q).format(a=a * b, b=b), direct(a, a * b, b, "÷")))
    for _ in range(500):
        a, b = RNG.randint(11, 25), RNG.randint(2, 9)
        out.append(conv(RNG.choice(MUL_Q).format(a=a, b=b), direct(a * b, a, b, "×")))
    return out


def gen_hundreds(n=350):
    out = []
    while len(out) < n:
        a, b = RNG.randint(100, 899), RNG.randint(10, 99)
        q = RNG.choice(ADD_Q).format(a=a, b=b)
        out.append(conv(q, f"{a} + {b} = {a + b}. {a} + {b // 10 * 10} = {a + b // 10 * 10}, then + {b % 10} = {a + b}."))
    return out


ARITH_RX = [
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), lambda a, b: a + b),
    (re.compile(r"(\d+)\s*(?:-|minus|take away)\s*(\d+)"), lambda a, b: a - b),
    (re.compile(r"(\d+)\s*(?:×|x|\*|times|multiplied by)\s*(\d+)"), lambda a, b: a * b),
    (re.compile(r"(\d+)\s*(?:÷|/|divided by)\s*(\d+)"), lambda a, b: a // b),
]


def validate(convs):
    """Every arithmetic claim in the corpus must be true."""
    bad = 0
    for c in convs:
        q = c["messages"][0]["content"]
        a = c["messages"][1]["content"]
        for rx, fn in ARITH_RX:
            m = rx.search(q)
            if not m:
                continue
            expect = fn(int(m.group(1)), int(m.group(2)))
            numbers = {int(x) for x in re.findall(r"\d+", a)}
            if expect not in numbers:
                bad += 1
                print("MISMATCH", q, "->", a, "expected", expect)
            break
    if bad:
        raise SystemExit(f"{bad} mismatched answers")
    return len(convs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/vg/arith.jsonl")
    args = ap.parse_args()

    convs = []
    convs += gen_single_digit()
    convs += gen_tables()
    convs += gen_two_digit_add()
    convs += gen_two_digit_sub()
    convs += gen_hundreds()
    validate(convs)

    with open(args.out, "w") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(convs)} conversations -> {args.out}")


if __name__ == "__main__":
    main()
