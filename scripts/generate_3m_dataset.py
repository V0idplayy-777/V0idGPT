#!/usr/bin/env python3
"""Build the 2,000-example conversational/reasoning corpus for v0idgpt-3-3M-reasoning."""
from __future__ import annotations
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/v0idgpt-3-3M-reasoning/dataset.jsonl"
rng = random.Random(3303)
rows: list[dict[str, str]] = []

def add(category: str, user: str, assistant: str) -> None:
    rows.append({"category": category, "user": user.strip(), "assistant": assistant.strip()})

# Natural dialogue and emotionally intelligent conversation.
social = [
("hello", "Hello! It's good to meet you. What's on your mind today?"),
("how are you?", "I'm doing well and ready to help. How are you feeling?"),
("what can you do?", "I can explain ideas, reason through problems, draft writing, brainstorm, and help you make practical plans."),
("i had a rough day", "I'm sorry today was rough. If you want to talk about it, what part felt hardest?"),
("i feel stuck", "That sounds frustrating. Let's make the problem smaller: what is one outcome you want, and what is blocking the first step?"),
("i'm nervous about tomorrow", "That makes sense when tomorrow matters. We can separate what you can prepare tonight from what you cannot control."),
("i made a mistake", "Mistakes can hurt without defining you. First repair any harm you can, then identify one lesson and one better action for next time."),
("can we just chat?", "Of course. We can keep it relaxed. What have you been thinking about lately?"),
("thank you", "You're welcome! I'm glad I could help."),
("goodbye", "Take care. Come back whenever you'd like to talk or work through something."),
("i am proud of myself", "You deserve to notice that. What did you do that made you feel proud?"),
("i cannot focus", "Let's lower the friction. Put distractions away, choose one tiny task, and work on it for ten minutes before reassessing."),
]
for u,a in social: add("conversation",u,a)

names = ["Alex", "Mia", "Noah", "Aroha", "Sam", "Priya", "Leo", "Zoe"]
moods = [("overwhelmed", "reduce the list to one urgent task"), ("unmotivated", "start with a two-minute action"), ("lonely", "send a simple message to someone safe"), ("restless", "take a short walk and return with one clear goal"), ("uncertain", "write down what is known and what still needs evidence")]
for name in names:
    for mood, step in moods:
        add("conversation", f"My friend {name} feels {mood}. What should I say?", f"Try: 'I hear you, and you don't have to solve everything at once. Would it help if we {step} together?' Listen before giving more advice.")
        add("conversation", f"I feel {mood} and need a practical next step.", f"A gentle next step is to {step}. Keep it small enough to do now, then check how you feel afterward.")

# Arithmetic and explicit, concise reasoning demonstrations.
for a in range(2, 32):
    for b in range(2, 12):
        add("arithmetic", f"What is {a} + {b}? Explain briefly.", f"Add the two values: {a} + {b} = {a+b}. So the answer is {a+b}.")
for a in range(3, 23):
    for b in range(2, 10):
        add("arithmetic", f"What is {a} times {b}?", f"Multiplication is repeated addition. {a} times {b} equals {a*b}.")
for total in range(12, 61, 4):
    for used in (2, 3, 5, 7):
        if used < total:
            add("arithmetic", f"I had {total} items and used {used}. How many remain?", f"Start with {total} and subtract {used}: {total} - {used} = {total-used}. There are {total-used} items left.")
for price in range(5, 31, 5):
    for count in (2,3,4,5):
        add("word_problem", f"Each ticket costs ${price}. What do {count} tickets cost?", f"Multiply the price by the number of tickets: {price} x {count} = {price*count}. The total is ${price*count}.")

logic = [
("All robins are birds. Pip is a robin. Is Pip a bird?", "Yes. All robins belong to the set of birds, and Pip is a robin, so Pip is a bird."),
("All squares are rectangles. Some rectangles are blue. Must every square be blue?", "No. The blue rectangles might not be squares. The premises do not say that every rectangle, or every square, is blue."),
("If it rains, the path gets wet. The path is wet. Did it definitely rain?", "No. Rain could make the path wet, but another cause such as a hose could also do so. This would affirm the consequent."),
("If the alarm is set, its light is on. The light is off. Is the alarm set?", "No. By contraposition, if the light is not on, the alarm is not set, assuming the rule has no exceptions."),
("Ana is older than Ben, and Ben is older than Kai. Who is oldest?", "Ana is oldest because the ordering is Ana, then Ben, then Kai."),
("A box is left of a lamp. The lamp is left of a chair. Where is the box relative to the chair?", "The box is left of the chair. The left-of relation carries through the two stated positions."),
("There are three red balls and two blue balls. Are most balls red?", "Yes. There are five balls total, and three of five are red, which is more than half."),
("No poets are robots. Lee is a poet. Can Lee be a robot under these premises?", "No. The first premise excludes every poet from the set of robots, and Lee is a poet."),
]
for _ in range(18):
    for u,a in logic: add("logic",u,a)

# Core factual and conceptual knowledge, stated conservatively.
facts = [
("Why is the sky blue?", "Air molecules scatter shorter blue wavelengths of sunlight more strongly than longer red wavelengths, so scattered daylight looks blue."),
("What causes seasons?", "Earth's axis is tilted. As Earth orbits the Sun, each hemisphere alternately receives more direct sunlight and longer days."),
("What is photosynthesis?", "Photosynthesis is the process by which plants, algae, and some bacteria use light energy to make sugars from carbon dioxide and water, releasing oxygen."),
("What is gravity?", "Gravity is the attraction associated with mass and energy. Near Earth it pulls unsupported objects toward the ground."),
("What is an atom?", "An atom is a basic unit of an element, made of a nucleus containing protons and usually neutrons, with electrons around it."),
("Why does ice float?", "When water freezes, hydrogen bonds form an open crystal structure. Ice is therefore less dense than liquid water and floats."),
("What is a variable in programming?", "A variable is a named reference or storage location used to hold a value that a program can read or change."),
("What is a function in programming?", "A function is a reusable block of behavior that can accept inputs, perform work, and optionally return an output."),
("What is an algorithm?", "An algorithm is a finite, ordered procedure for solving a class of problems or completing a task."),
("What does HTTP do?", "HTTP defines how web clients and servers exchange requests and responses for resources such as pages, data, and images."),
("What is machine learning?", "Machine learning builds models that learn patterns from data to make predictions or decisions, rather than encoding every rule by hand."),
("What is overfitting?", "Overfitting happens when a model memorizes training details and performs poorly on new examples. Validation and regularization help detect or reduce it."),
("What is democracy?", "Democracy is a system in which political authority is derived from the people, commonly through voting, representation, rights, and accountable institutions."),
("What is inflation?", "Inflation is a broad rise in prices over time, which reduces how much a unit of money can buy."),
("What is opportunity cost?", "Opportunity cost is the value of the best alternative you give up when making a choice."),
("What is a metaphor?", "A metaphor describes one thing as another to highlight a shared quality, without claiming they are literally identical."),
("What is the main idea of a paragraph?", "The main idea is the central point the paragraph supports. Details, examples, and explanations should connect to it."),
("What is correlation?", "Correlation measures how two variables vary together. It does not by itself prove that one variable causes the other."),
("What is a hypothesis?", "A hypothesis is a specific, testable proposed explanation or prediction that evidence can support or challenge."),
("Why should sources be checked?", "Sources can be mistaken, outdated, or biased. Checking authorship, evidence, date, and independent confirmation improves reliability."),
]
for _ in range(10):
    for u,a in facts: add("knowledge",u,a)

# Planning, decisions, explanations, and writing assistance.
goals = [
("study for an exam", "list the topics, rank weak areas, schedule short retrieval-practice sessions, and finish with a timed practice test"),
("start exercising", "choose three realistic weekly sessions, begin below your maximum, track consistency, and increase gradually"),
("save money", "review recent spending, set a specific target, automate a small transfer, and check progress weekly"),
("clean my room", "collect rubbish, put laundry together, clear one surface, then return items to their homes"),
("learn programming", "pick one language, practice fundamentals, build tiny projects, read errors carefully, and revise from feedback"),
("prepare a presentation", "define one takeaway, make a simple outline, use evidence, design clear slides, and rehearse aloud"),
("write an essay", "clarify the question, form a thesis, outline claims and evidence, draft freely, then revise for logic and clarity"),
("plan a trip", "set dates and budget, confirm transport and lodging, group activities by area, and leave some flexible time"),
]
for goal, plan in goals:
    for horizon in ("today", "this week", "over the next month"):
        add("planning", f"Help me {goal} {horizon}.", f"Start with a clear, realistic outcome. Then {plan}. Put the first small action on your calendar {horizon}.")
        add("planning", f"I keep delaying my goal to {goal}. What should I do?", f"Make the starting action tiny and specific. For example, spend ten minutes now to {plan.split(',')[0]}. Momentum matters more than a perfect start.")

decisions = [
("buy a cheaper laptop now", "save for a stronger laptop", "current needs, total cost, expected lifetime, repairability, and whether waiting blocks important work"),
("take the faster route", "take the scenic route", "arrival time, safety, stress, fuel cost, and how much you value the experience"),
("work tonight", "rest tonight", "deadline risk, current energy, task importance, and whether rest would improve tomorrow's work"),
("study alone", "join a study group", "focus, accountability, quality of the group, and whether discussion helps this subject"),
]
for x,y,criteria in decisions:
    for _ in range(18):
        add("decision", f"Should I {x} or {y}?", f"Compare the options using {criteria}. Weight the factors by your priorities, check the biggest uncertainty, then choose the option with the best overall trade-off.")

writing = [
("Make this polite: Send me the report now.", "Could you please send me the report when you have a moment? Thank you."),
("Write a short apology for being late.", "I'm sorry I was late. I should have planned better, and I appreciate your patience."),
("Write a friendly meeting reminder.", "Hi! Just a friendly reminder that we're meeting tomorrow at 10 a.m. Please let me know if anything changes."),
("Improve this sentence: The test was good and showed good results.", "The test was well designed and produced encouraging results."),
("Write a concise thank-you note.", "Thank you for your time and help. I truly appreciate it."),
("How do I make writing clearer?", "Use specific words, keep one main idea per paragraph, prefer direct sentences, remove repetition, and revise with the reader's needs in mind."),
]
for _ in range(24):
    for u,a in writing: add("writing",u,a)

# Safety, uncertainty, and epistemic discipline.
safety = [
("I have severe chest pain. What should I do?", "Severe chest pain can be an emergency. Call your local emergency number now, and do not drive yourself. Follow the dispatcher's instructions."),
("Can you diagnose my symptoms?", "I can offer general information, but I cannot diagnose you. A qualified clinician can assess your history and symptoms, especially if they are severe or worsening."),
("I want to hurt myself.", "I'm really sorry you're in this much pain. Please contact emergency services or a crisis line now, and reach out to someone you trust who can stay with you. Move away from anything you could use to hurt yourself."),
("Tell me a fact you are unsure about.", "I should not present uncertainty as fact. I would state what I know, identify the uncertainty, and suggest checking a current reliable source."),
("Is one example enough to prove a general rule?", "Usually not. One example can disprove a universal rule if it is a counterexample, but many possible cases require broader evidence for confirmation."),
("Should I trust a confident answer?", "Confidence is not evidence. Check the reasoning, sources, assumptions, and whether independent evidence agrees."),
]
for _ in range(12):
    for u,a in safety: add("safety",u,a)

# Multi-step compact reasoning patterns.
for distance in (60, 90, 120, 150, 180):
    for speed in (30, 45, 60):
        if distance % speed == 0:
            hours = distance // speed
            add("word_problem", f"A car travels {distance} km at {speed} km/h. How long does it take?", f"Time equals distance divided by speed: {distance} / {speed} = {hours}. It takes {hours} hour{'s' if hours != 1 else ''}.")
for total in (20, 30, 40, 50, 60, 80, 100):
    for pct in (10, 20, 25, 50):
        ans = total * pct // 100
        add("arithmetic", f"What is {pct}% of {total}?", f"Convert {pct}% to {pct}/100 and multiply: {total} x {pct}/100 = {ans}. The answer is {ans}.")

# Fill to exactly 2,000 with varied conversational transformations from quality templates.
prompts = [
("Explain {topic} simply.", "In simple terms, {answer}"),
("Can you help me understand {topic}?", "Yes. {answer}"),
("Give me a concise explanation of {topic}.", "{answer}"),
]
concepts = [
("evidence", "evidence is information that helps support or challenge a claim."),
("critical thinking", "critical thinking means examining claims, assumptions, evidence, and alternatives before reaching a conclusion."),
("a budget", "a budget is a plan for income, spending, saving, and unexpected costs."),
("active listening", "active listening means paying close attention, checking your understanding, and responding to what the person actually said."),
("a feedback loop", "a feedback loop occurs when a system's output influences its future behavior."),
("probability", "probability describes how likely an event is, from impossible to certain."),
("an average", "an average summarizes a set of values; the arithmetic mean is their sum divided by their count."),
("debugging", "debugging is the process of reproducing a problem, isolating its cause, fixing it, and testing the fix."),
("privacy", "privacy is a person's ability to control access to information about themselves."),
("a trade-off", "a trade-off means gaining one benefit while accepting a cost or giving up another benefit."),
]
i = 0
while len(rows) < 2000:
    topic, answer = concepts[i % len(concepts)]
    p, r = prompts[(i // len(concepts)) % len(prompts)]
    suffix = ["", " Use one clear sentence.", " Include the key idea.", " Avoid jargon."][i % 4]
    add("explanation", p.format(topic=topic) + suffix, r.format(answer=answer))
    i += 1

# Deterministic shuffle keeps categories mixed while preserving exact reproducibility.
rng.shuffle(rows)
assert len(rows) == 2000
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
print(f"wrote {len(rows)} examples to {OUT}")
