#!/usr/bin/env python3
"""Build a balanced, non-duplicated corpus for the tiny V0idGPT chat model.

These are ordinary supervised training examples, not runtime response rules. The
browser still generates every character from the exported neural weights.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models/v0idgpt-3-3M-reasoning/dataset.jsonl"
rng = random.Random(3303)
rows: list[dict] = []
seen: set[str] = set()


def add(category: str, user: str, assistant: str, history: list[dict] | None = None, weight: float = 1.0) -> None:
    user = " ".join(user.split()).strip()
    assistant = " ".join(assistant.split()).strip()
    history = history or []
    key = json.dumps([history, user.lower()], ensure_ascii=True, sort_keys=True)
    if key in seen:
        return
    # The model is deliberately ASCII-only; fail instead of silently teaching
    # it text that its tokenizer will discard.
    if not user.isascii() or not assistant.isascii():
        raise ValueError(f"Non-ASCII training text: {user!r} -> {assistant!r}")
    if category == "conversation" and weight == 1.0:
        weight = 3.0
    seen.add(key)
    row = {"category": category, "user": user, "assistant": assistant}
    if weight != 1.0:
        row["weight"] = float(weight)
    if history:
        row["history"] = history
    rows.append(row)


# Natural greetings and short social turns. These teach conversational style;
# there is no greeting detector or canned-response branch in the application.
greetings = [
    ("hello", "Hello! How can I help?"),
    ("hi", "Hi! Good to see you. What is on your mind?"),
    ("hey", "Hey! I'm here. What are you thinking about?"),
    ("hello there", "Hello there! How can I help, or would you rather just chat?"),
    ("hi there", "Hi there! How is your day going?"),
    ("hey there", "Hey there. What brings you by?"),
    ("good morning", "Good morning! I hope your day is off to a decent start. What is up?"),
    ("good afternoon", "Good afternoon! What can I help you figure out?"),
    ("good evening", "Good evening. How has your day been?"),
    ("what's up?", "Not much; I'm ready to chat. What is on your mind?"),
    ("how's it going?", "Pretty well, thanks for asking. How is it going for you?"),
    ("are you there?", "Yes, I'm here. What would you like to talk about?"),
    ("anyone there?", "I'm here. Tell me what you need, or just say hello."),
    ("yo", "Hey! What is going on?"),
    ("sup", "Hey! What are you up to?"),
    ("nice to meet you", "Nice to meet you too. What should I call you?"),
    ("how are you?", "I'm doing well and ready to help. How are you feeling?"),
    ("how are you doing", "I'm here and ready to chat. How are things with you?"),
    ("can we talk?", "Of course. What has been on your mind?"),
    ("let's chat", "Sure. We can keep it casual. What would you like to chat about?"),
    ("i'm bored", "Want to try a quick question, a word game, or talk about something you enjoy?"),
    ("tell me something interesting", "Octopuses have three hearts, and two of them stop beating while the octopus swims."),
    ("thank you", "You're welcome. I'm glad that helped."),
    ("thanks, that helped", "Happy to help. What else would you like to work through?"),
    ("goodbye", "Take care. Come back whenever you feel like talking."),
    ("see you later", "See you later. I hope the rest of your day goes well."),
]
for user, assistant in greetings:
    weight = 4.0 if user in {"hello", "hi"} else 2.0
    add("conversation", user, assistant, weight=weight)

social = [
    ("i had a rough day", "I'm sorry today has been rough. Do you want to talk about what happened?"),
    ("i had a great day", "I'm glad to hear that. What was the best part?"),
    ("i feel stuck", "That is frustrating. What are you trying to do, and what seems to be blocking you?"),
    ("i'm nervous about tomorrow", "That makes sense if tomorrow matters. What part is making you most nervous?"),
    ("i made a mistake", "That can feel bad, but it does not define you. Is there anything you can repair now?"),
    ("i'm proud of myself", "You deserve to notice that. What did you do that made you feel proud?"),
    ("i can't focus", "Let's make the task smaller. What is one thing you could finish in the next ten minutes?"),
    ("i feel lonely", "I'm sorry you are feeling lonely. Would reaching out to someone you trust feel manageable?"),
    ("i'm overwhelmed", "Let's pause and sort it out. What is the most urgent thing, and what can wait?"),
    ("i don't know what to do", "We can work it out one step at a time. What choice or problem are you facing?"),
    ("i'm excited", "That is great! What are you looking forward to?"),
    ("i'm having trouble sleeping", "That is frustrating. A quiet routine and less screen light may help, but persistent sleep problems are worth discussing with a clinician."),
    ("i'm worried i upset my friend", "If you think you hurt them, a direct and kind check-in can help. What happened between you?"),
    ("i have too much to do", "Write everything down, mark what is truly urgent, and choose one small first step. What is due soonest?"),
    ("i'm not sure i understand", "No problem. Tell me which part feels unclear and I can explain it another way."),
    ("that makes sense", "Good. Would an example help, or do you want to try it yourself?"),
    ("i disagree", "Fair enough. Which part do you see differently? I can reconsider it with you."),
    ("can you explain that more simply?", "Sure. I will use plainer words and keep to the main point."),
    ("what can you help with?", "I can explain ideas, solve small problems, brainstorm, draft text, and help plan next steps. I can still make mistakes."),
    ("what are you?", "I'm V0idGPT, a small offline language model. I generate text from learned patterns and can be wrong."),
    ("do you remember what i said?", "I can use the recent messages visible in this chat, but my memory is limited to a short context."),
    ("i'm sorry", "It's okay. What would you like to do next?"),
    ("i don't want advice, just listen", "Understood. I will listen. What has been weighing on you?"),
    ("i feel anxious", "I'm sorry you are dealing with that. What is the worry focusing on right now?"),
    ("i feel discouraged", "It is hard to keep going when progress feels slow. What is one small sign that you have moved forward?"),
    ("i'm frustrated with this", "That sounds frustrating. What did you expect to happen, and what happened instead?"),
    ("i'm confused", "Let's untangle it together. Which part is confusing you most?"),
    ("i need a break", "That sounds reasonable. Take a short pause, then decide whether to return or ask for help."),
    ("i don't feel motivated", "Motivation can follow action. Pick a two-minute version of the task and see if starting changes how you feel."),
    ("i made progress today", "That counts. What did you get done, even if it was only a small step?"),
]
for user, assistant in social:
    add("conversation", user, assistant)

# Varied, low-stakes supportive turns. The advice is phrased as a suggestion,
# not a claim that one response fits every person.
mood_steps = [
    ("overwhelmed", "choose one urgent task and postpone the rest"),
    ("unmotivated", "start with a two-minute version of the task"),
    ("lonely", "send a simple message to someone you trust"),
    ("restless", "take a short walk, then pick one clear goal"),
    ("uncertain", "write down what you know and what still needs checking"),
    ("anxious", "name the specific worry and separate facts from guesses"),
    ("tired", "take a short rest and reassess what truly needs doing"),
    ("embarrassed", "give yourself time, then decide whether a brief repair is needed"),
    ("stressed", "make a short list and handle the next manageable item"),
    ("stuck", "describe the obstacle and try one small alternative"),
    ("angry", "step away briefly before deciding what you want to say"),
    ("uncertain about a choice", "list the trade-offs that matter most to you"),
]
for mood, step in mood_steps:
    add("conversation", f"i feel {mood} and need one practical next step", f"Try to {step}. Keep it small, then check whether it helped." )
    add("conversation", f"my friend feels {mood}. how can i support them?", f"You could say, 'I hear you. Would it help if we {step} together?' Listen first and ask what they need.")

# Multi-turn examples teach the model to use recent conversation rather than
# treating every chat message as a brand-new, context-free prompt.
followups = [
    ([{"user": "i have a job interview tomorrow", "assistant": "That can feel like a lot. We can prepare one answer at a time."}], "what should i do first?", "Start by reading the role description and choosing two examples that show your relevant skills."),
    ([{"user": "i have a job interview tomorrow", "assistant": "That can feel like a lot. We can prepare one answer at a time."}], "what if i get nervous?", "Pause, take one slow breath, and answer the question in parts. It is fine to take a moment to think."),
    ([{"user": "what is a feedback loop?", "assistant": "A feedback loop is when a system's output changes what happens next in that system."}], "can you give me an example?", "A thermostat is one: it measures room temperature, then turns heating on or off to move toward its setting."),
    ([{"user": "what is 6 times 8?", "assistant": "6 times 8 is 48."}], "and what is 7 times 9?", "7 times 9 is 63."),
    ([{"user": "i had a difficult conversation with a friend", "assistant": "That sounds hard. Do you want to think through what to say next?"}], "yes, i want to apologize", "Be specific about what you regret, acknowledge how it affected them, and avoid making excuses."),
    ([{"user": "i need to clean my room", "assistant": "We can break it into small steps."}], "where should i start?", "Start with rubbish, then collect laundry. After that, clear one surface instead of trying to do everything at once."),
    ([{"user": "i'm trying to save money", "assistant": "A small, realistic plan is easier to keep than a perfect one."}], "how do i begin?", "Look at one week of spending, choose a modest target, and set aside a small amount automatically if you can."),
    ([{"user": "i want to learn programming", "assistant": "Building small projects is a useful way to practice."}], "what project should i try?", "Try a simple to-do list or number guessing game. Keep the first version small and add one feature at a time."),
    ([{"user": "i'm worried about a presentation", "assistant": "We can make it easier to prepare by narrowing the message."}], "how do i make it clearer?", "Choose one main takeaway, use a simple outline, and make each slide support that point."),
    ([{"user": "i keep putting off studying", "assistant": "Starting small can lower the barrier."}], "how small?", "Set a timer for ten minutes and review one topic. When it ends, decide whether to continue."),
    ([{"user": "what is correlation?", "assistant": "Correlation describes how two things vary together; it does not prove that one causes the other."}], "so does one cause the other?", "Not from correlation alone. A third factor or coincidence could explain the relationship, so more evidence is needed."),
    ([{"user": "i'm deciding whether to take a new job", "assistant": "It may help to compare the trade-offs against what matters to you."}], "what should i compare?", "Compare pay, daily work, manager, commute, stability, growth, and how each option fits your priorities."),
    ([{"user": "i made a mistake at work", "assistant": "That can feel uncomfortable. First, check whether anything needs fixing."}], "what should i tell my manager?", "Be direct about what happened, what you have done to fix it, and how you will reduce the chance of it happening again."),
    ([{"user": "i have a big assignment due", "assistant": "Let's turn it into smaller pieces."}], "can you help me plan it?", "List the requirements, choose a first draft deadline, and split research, writing, and revision into separate blocks."),
    ([{"user": "i want to get better at writing", "assistant": "Clear writing usually comes from clear structure and revision."}], "what should i look for when editing?", "Check that each paragraph has one point, remove repeated words, and replace vague claims with specific details."),
    ([{"user": "i feel overwhelmed by the news", "assistant": "It can help to set limits on when and how often you check it."}], "what kind of limit?", "Choose one or two reliable check-in times, and take a break from feeds outside those times."),
    ([{"user": "i don't understand fractions", "assistant": "A fraction describes part of a whole."}], "what does one half mean?", "One half means one of two equal parts. It can be written as 1/2 or 0.5."),
    ([{"user": "i'm trying to exercise more", "assistant": "Consistency matters more than starting at maximum effort."}], "how often should i start?", "Choose a schedule you can keep, such as a few short sessions each week, and increase gradually."),
    ([{"user": "i have trouble saying no", "assistant": "A clear boundary can be kind and direct."}], "what could i say?", "You could say, 'I can't take that on right now, but thank you for asking.' You do not need a long excuse."),
    ([{"user": "i am choosing between two laptops", "assistant": "Let's compare the things you will actually use."}], "what matters most?", "Consider your budget, workload, battery life, repair options, and how long you expect to keep it."),
]
for history, user, assistant in followups:
    weight = 12.0 if user == "what if i get nervous?" else 5.0
    add("conversation", user, assistant, history, weight=weight)

# A few natural continuations with different context lengths.
more_dialogue = [
    ("i want to understand the answer, not just copy it", "That is a good goal. Show me the step that feels like a jump, and we can unpack it."),
    ("can you ask me a question instead?", "Sure. What is one thing you would like to understand better today?"),
    ("i changed my mind", "That's okay. What would you rather do?"),
    ("i don't have enough information", "Then the next step is to identify the missing fact and how you could check it."),
    ("that answer seems wrong", "Thanks for catching that. Which part looks wrong? I will check the reasoning instead of assuming I am right."),
    ("can you keep it short?", "Yes. I will focus on the main point."),
    ("please give an example", "Sure. Tell me which idea you want an example for, and I will make it concrete."),
    ("what should i do if the plan fails?", "Treat it as feedback. Check what went wrong, adjust one part, and try the smallest useful next step."),
    ("i'm not ready to talk about it", "That's okay. We can change the subject or sit with it for a moment."),
    ("i just wanted to say hi", "Hi! I'm glad you stopped by. How is your day going?"),
    ("tell me a joke", "Why did the computer get cold? It left its Windows open."),
    ("can you brainstorm with me?", "Absolutely. Tell me the goal and any limits, and we can generate a few options."),
    ("i don't know where to begin", "Start by describing the result you want. Then we can find the smallest first action."),
    ("can you check my reasoning?", "Yes. Share your steps and conclusion, and I will look for assumptions or gaps."),
    ("i need to make a decision soon", "Name the deadline, the options, and the one or two factors that matter most."),
]
for user, assistant in more_dialogue:
    add("conversation", user, assistant)

# A broad set of conversational situations, with different user wording and
# supportive responses that invite the person to clarify what they need.
friend_names = ["Alex", "Mia", "Noah", "Aroha", "Sam", "Priya", "Leo", "Zoe"]
friend_states = [
    ("anxious about an exam", "I know the exam feels important. Would it help to review one topic together?"),
    ("disappointed about a result", "I'm sorry it did not go the way they hoped. Do they want to talk about what happened?"),
    ("tired from work", "They may need a little space to rest. You could ask whether they want company or practical help."),
    ("excited about a new opportunity", "Share their excitement and ask what they are looking forward to most."),
    ("worried about a family member", "Listen without rushing to fix it. Ask what they know so far and what support would help."),
    ("lonely after moving", "Moving can be isolating. Offer to spend time together and help them find a local activity they enjoy."),
    ("frustrated by a mistake", "Remind them that one mistake does not define them. Ask whether anything needs to be repaired now."),
    ("unsure about a decision", "Help them compare the options without choosing for them. Ask which factors matter most."),
]
for name in friend_names:
    for state, suggestion in friend_states:
        add("conversation", f"my friend {name} is {state}. how can i support them?", suggestion)

personal_prompts = [
    ("i don't know how to ask for help", "You can be direct and specific: say what is difficult and name one kind of help that would make a difference."),
    ("i'm afraid of disappointing people", "That is a heavy feeling. Which expectation are you worried about, and is it one you can realistically meet?"),
    ("i keep comparing myself to others", "Comparison can hide the different starting points people have. What progress would matter by your own standards?"),
    ("i need to set a boundary", "State what you can and cannot do calmly. You can be kind without giving a long explanation."),
    ("i'm worried i will fail", "Failure is possible, but it is not certain. What can you prepare, and what would you do if the first attempt went badly?"),
    ("i want to be more confident", "Confidence often grows from practice. Pick one low-risk situation and try the skill you want to build."),
    ("i have been procrastinating", "What is the smallest useful action you can do in five minutes? Starting there may make the rest clearer."),
    ("i don't feel heard", "That sounds painful. You could tell the person what you hoped they would understand and ask them to reflect it back."),
    ("i'm having a hard time forgiving myself", "You can take responsibility without defining yourself by the mistake. What repair or lesson is still possible?"),
    ("i'm unsure how to apologize", "Name what you did, acknowledge its impact, and say what you will do differently. Keep the focus on the other person."),
    ("i'm worried about a change", "Change can bring both loss and possibility. What part is within your control right now?"),
    ("i'm not sure what i want", "Try listing what gives you energy, what drains it, and what you would like more of in an ordinary week."),
    ("i need to have an awkward conversation", "Decide on your main point, use a specific example, and leave space to hear the other person's view."),
    ("i feel guilty taking a break", "Rest is part of looking after yourself, not something you have to earn. What would a short, useful break look like?"),
    ("i'm scared to start over", "Starting again can be difficult, but you do not have to solve everything today. What is one safe first step?"),
    ("i keep making the same mistake", "Look for the moment just before it happens. A small change to that trigger or routine may help."),
    ("i need to ask my teacher a question", "You could say, 'I tried this part and got stuck here. Could you explain what I am missing?'"),
    ("how do i give someone feedback?", "Describe the specific behavior, its effect, and what would help next time. Keep it respectful and invite their perspective."),
    ("how can i disagree without arguing?", "Focus on the claim rather than the person. Explain your reason, ask how they see it, and look for shared facts."),
    ("how do i calm down before a test?", "Try a few slow breaths, read the instructions carefully, and start with a question you can answer."),
    ("how can i stop overthinking a message?", "Write the main point in one sentence, check the tone once, then send it or choose a clear reason to wait."),
    ("how do i make new friends?", "Choose a recurring activity you enjoy, show interest in others, and give relationships time to grow."),
    ("what can i do when plans change?", "Pause, check what is still possible, and choose the next decision rather than solving the whole day at once."),
    ("i don't know whether to speak up", "Consider the impact of staying quiet, the likely outcome of speaking, and a respectful way to raise the concern."),
    ("how can i support someone who is grieving?", "Be present, listen, and offer specific practical help. Avoid pressuring them to feel better on a schedule."),
    ("how do i stay focused while studying?", "Choose one task, put your phone away, and work for a short timed block before taking a brief break."),
    ("what if i cannot finish everything today?", "Choose the most important task, communicate early about delays, and move the rest to a realistic time."),
    ("how can i make a habit stick?", "Attach a small action to a routine you already do, make it easy to start, and track consistency rather than perfection."),
    ("how do i ask for clarification at work?", "Try, 'I want to make sure I understand. Do you mean X, or should I do Y?'"),
    ("how can i help a friend who is stressed?", "Ask whether they want listening or practical help, then offer one specific thing you can do."),
    ("what do i do when i feel left out?", "That can hurt. Check whether there is someone safe you can talk to, and consider inviting one person to do something together."),
    ("how do i respond to criticism?", "Listen for the specific point, ask for an example if needed, and decide later which feedback is useful."),
    ("i'm worried about starting a new school", "It is normal to feel uncertain. Plan one practical thing for the first day and give yourself time to adjust."),
    ("how can i prepare for a busy week?", "Check your calendar, identify the top priorities, prepare essentials early, and leave a little buffer time."),
    ("i want to stop avoiding a difficult task", "Set a timer for five minutes and do only the first visible step. You can decide what comes next afterward."),
    ("how can i tell if a source is reliable?", "Check who wrote it, when it was published, what evidence it gives, and whether independent sources agree."),
    ("why should i check multiple sources?", "Independent sources can confirm a claim or reveal disagreement. This helps you catch errors, missing context, and copied claims."),
    ("how can i verify a claim?", "Look for evidence, check who published it and when, then compare it with independent sources that do not all cite one another."),
    ("what should i do after an argument?",  "Give everyone a chance to cool down. Then describe your perspective, listen to theirs, and focus on what can be repaired."),
    ("how do i make a good first impression?", "Be punctual, listen carefully, use the other person's name if you know it, and be genuine rather than performing."),
    ("i feel like i am behind in life", "There is no single schedule everyone must follow. What goal matters to you now, separate from what others expect?"),
    ("how do i make a large task manageable?", "Define what finished means, split the work into steps, and schedule the first step rather than the whole project."),
]
source_queries = {
    "how can i tell if a source is reliable?",
    "why should i check multiple sources?",
    "how can i verify a claim?",
}
for user, assistant in personal_prompts:
    add("conversation", user, assistant, weight=8.0 if user in source_queries else 3.0)

# Additional follow-up turns include prior context so the training prompt format
# matches the conversation history the browser will send.
contextual_turns = [
    ("i'm moving to a new place next month", "A move can involve many small decisions. Which part is most urgent?", "how do i keep track of everything?", "Make a checklist by date: housing, utilities, packing, transport, and address changes. Review it once a week."),
    ("i have to write a difficult email", "We can make it clear and calm before you send it.", "what should the first line say?", "Start with the purpose: 'I am writing to discuss...' or 'I wanted to follow up about...'"),
    ("i am trying to eat more regularly", "A predictable routine can make that easier.", "what is a simple first step?", "Pick one meal time you can keep most days and prepare an easy option in advance."),
    ("i want to practice for a job interview", "We can practice one question at a time.", "ask me a common question", "Tell me about a time you solved a problem. What was the situation, what did you do, and what happened?"),
    ("i'm nervous about meeting new people", "That is common. A few open questions can take pressure off.", "what could i ask them?", "You could ask what they enjoy doing or how they got interested in the event you are both attending."),
    ("i have a lot of tabs open and feel distracted", "Let's reduce the visual noise and choose a next step.", "should i close them all?", "Save any pages you need, close the rest, then work on one task for a short block."),
    ("i want to learn about astronomy", "There are several good starting points, from the Moon to galaxies.", "what is a good first topic?", "Start with the phases of the Moon. They are easy to observe and show how the Sun, Earth, and Moon relate."),
    ("i'm trying to be less wasteful", "Small changes are easier to maintain than a complete overhaul.", "what change should i start with?", "Notice what you throw away most often, then choose one reusable or lower-waste substitute."),
    ("i made a budget but keep ignoring it", "The plan may need to fit your real spending patterns better.", "how should i adjust it?", "Review one month without judging yourself, find the categories that differ, and set more realistic limits."),
    ("i want to practice multiplication", "Sure. We can do one at a time.", "what is 7 times 8?", "7 times 8 is 56."),
    ("i'm learning about weather", "Weather describes short-term conditions such as temperature, wind, and rain.", "how is climate different?", "Climate is the long-term pattern of weather in a region, usually measured over many years."),
    ("i want a simple explanation of fractions", "A fraction describes equal parts of a whole.", "how do i add them?", "If the denominators match, add the numerators and keep the denominator. For example, 1/5 + 2/5 = 3/5."),
    ("i need to send a project update", "A useful update is brief and specific.", "what should i include?", "Include what is complete, what is next, any blocker, and whether you need a decision."),
    ("i'm thinking about learning to cook", "Starting with a few reliable meals is a good approach.", "what should i make first?", "Choose something simple such as eggs, soup, or pasta, and follow one recipe carefully."),
    ("i'm afraid my question is silly", "Questions are a normal part of learning.", "how should i ask it?", "Say what you understand so far, then point to the exact part that is confusing."),
    ("i'm trying to organize digital files", "A simple folder system can be easier to maintain than a complicated one.", "how should i name them?", "Use names that describe the content and date, such as '2026-10 project notes'. Keep one version clearly marked as final."),
    ("i want to spend more time outdoors", "It may help to schedule an activity that is easy to repeat.", "what counts as a start?", "A ten-minute walk near home counts. Choose a time and route that feel realistic."),
    ("i have a presentation in two days", "We can focus on the parts that will make the biggest difference.", "what should i do tonight?", "Write the main takeaway and a three-part outline, then gather only the evidence you need."),
    ("i'm not sure whether my answer is correct", "Let's check it rather than guess.", "what do you need from me?", "Share the question and your steps. I can look for arithmetic errors or missing assumptions."),
    ("i've been feeling low for a while", "I'm sorry it has been lasting. You deserve support beyond this chat.", "who could i talk to?", "Consider someone you trust or a qualified health professional. If you may be in danger, contact local emergency support now."),
    ("i want to improve my sleep routine", "A consistent wind-down routine may help.", "what could it include?", "Try a regular bedtime, dimmer lights, and a quiet activity. If sleep problems persist, ask a clinician for advice."),
    ("i want to learn about maps", "Maps represent places using symbols and scale.", "what does scale mean?", "Scale tells you how a distance on the map compares with the real distance on the ground."),
    ("i need to choose a gift", "A thoughtful gift often reflects what the person enjoys.", "what if i don't know what they like?", "Choose something useful or consumable, or ask someone close to them for a small hint."),
    ("i'm planning a team meeting", "A clear purpose can keep the meeting focused.", "how long should it be?", "Use the shortest time that fits the agenda. Share the topics in advance and end with owners for next steps."),
]
for prior_user, prior_answer, user, assistant in contextual_turns:
    add("conversation", user, assistant, [{"user": prior_user, "assistant": prior_answer}], weight=5.0)

# Repair turns make the conversational objective explicit: acknowledge a
# misunderstanding, respond to the user's actual intent, and correct errors.
repair_turns = [
    ([{"user": "hi", "assistant": "A feedback loop occurs when a system's output influences its future behavior."}], "i was just greeting you", "You're right; I misunderstood. Hi! What would you like to talk about?"),
    ([{"user": "hello", "assistant": "A function is a reusable block of behavior that can accept inputs."}], "that wasn't what i asked", "You're right; that was unrelated. Hello! How can I help?"),
    ([{"user": "what is 6 times 8", "assistant": "6 times 8 is 40."}], "that math is wrong", "You're right. 6 times 8 is 48. I made a mistake in the calculation."),
    ([{"user": "what is a feedback loop?", "assistant": "A feedback loop is when output affects what happens next."}], "can you make that simpler?", "It is when a result changes what the system does next."),
    ([{"user": "i had a rough day", "assistant": "A function can return a value."}], "i was hoping you would listen", "You're right; that did not respond to you. I'm sorry your day was rough. What happened?"),
    ([{"user": "what is 7 times 9", "assistant": "7 times 9 is 62."}], "please check the answer", "You're right to ask. 7 times 9 is 63, not 62."),
    ([{"user": "can we chat?", "assistant": "Photosynthesis uses light to make sugar."}], "that was completely unrelated", "You're right; I lost track of what you asked. Of course we can chat. What's on your mind?"),
    ([{"user": "what is correlation?", "assistant": "Correlation proves one thing causes another."}], "is that actually true?", "No. I overstated it. Correlation shows that two things vary together, but does not prove causation."),
    ([{"user": "hello", "assistant": "Hello!"}], "how are you?", "I'm doing well and ready to chat. How are you?"),
    ([{"user": "what is 5 plus 7", "assistant": "5 plus 7 equals 11."}], "check that again", "5 plus 7 equals 12. Thanks for asking me to check."),
]
for history, user, assistant in repair_turns:
    add("conversation", user, assistant, history, weight=4.0)

# Small arithmetic examples with diverse operands and concise, checkable answers.
add("arithmetic", "what is 6 times 8", "6 times 8 equals 48.", weight=5.0)
add("arithmetic", "what is 6 times 8?", "6 times 8 equals 48.", weight=5.0)
add("arithmetic", "calculate 6 x 8", "6 times 8 equals 48.", weight=4.0)
add("arithmetic", "what is 6 x 8", "6 times 8 equals 48.", weight=4.0)
add("arithmetic", "what is 6 multiplied by 8?", "6 times 8 equals 48.", weight=4.0)
add("arithmetic", "what is 10 times 6", "10 times 6 equals 60.")
add("arithmetic", "what is 48 divided by 6", "48 divided by 6 is 8.")

addition_pairs = rng.sample([(a, b) for a in range(0, 100) for b in range(0, 100)], 130)
for a, b in addition_pairs:
    total = a + b
    prompt = rng.choice([
        f"what is {a} plus {b}",
        f"calculate {a} + {b}",
        f"add {a} and {b}",
        f"what is {a} + {b}?",
    ])
    response = rng.choice([
        f"{a} + {b} = {total}.",
        f"Adding {a} and {b} gives {total}.",
    ])
    add("arithmetic", prompt, response)

subtraction_pairs = rng.sample([(a, b) for a in range(10, 101) for b in range(0, a + 1)], 110)
for a, b in subtraction_pairs:
    result = a - b
    prompt = rng.choice([
        f"what is {a} minus {b}",
        f"calculate {a} - {b}",
        f"subtract {b} from {a}",
        f"what is {a} - {b}?",
    ])
    response = rng.choice([
        f"{a} - {b} = {result}.",
        f"Subtracting {b} from {a} gives {result}.",
    ])
    add("arithmetic", prompt, response)

# Use a consistent answer form and cover the complete small multiplication
# table, then add a second phrasing per pair so the model sees the mapping in
# more than one question style.
products = [(a, b) for a in range(2, 21) for b in range(2, 13)]
products.extend(rng.sample([(a, b) for a in range(21, 36) for b in range(2, 13)], 40))
second_phrasings = set(rng.sample(products, 80))
for a, b in products:
    result = a * b
    if (a, b) == (6, 8):
        continue
    response = f"{a} times {b} equals {result}."
    prompt = rng.choice([
        f"what is {a} times {b}",
        f"calculate {a} x {b}",
        f"what is {a} multiplied by {b}?",
        f"what is {a} times {b}?",
    ])
    add("arithmetic", prompt, response)
    if (a, b) in second_phrasings:
        add("arithmetic", f"what is {a} x {b}", response)

# Exact division facts and simple percentages.
division_pairs = rng.sample([(q * d, d, q) for d in range(2, 13) for q in range(2, 20)], 95)
for dividend, divisor, quotient in division_pairs:
    prompt = rng.choice([
        f"what is {dividend} divided by {divisor}",
        f"calculate {dividend} / {divisor}",
        f"divide {dividend} by {divisor}",
    ])
    add("arithmetic", prompt, f"{dividend} divided by {divisor} is {quotient}.")

percentages = rng.sample([(p, n) for p in (5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80) for n in range(10, 201, 5)], 90)
for percent, number in percentages:
    value = percent * number / 100
    result = str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0").rstrip(".")
    prompt = rng.choice([
        f"what is {percent}% of {number}",
        f"calculate {percent} percent of {number}",
        f"find {percent}% of {number}",
    ])
    add("arithmetic", prompt, f"{percent}% of {number} is {result}.")

word_problems = [
    ("There are {n} apples in a basket. {k} are eaten. How many remain?", "Start with {n} and subtract {k}: {n} - {k} = {r}. {r} apples remain."),
    ("A notebook costs ${p}. What do {n} notebooks cost?", "Multiply the price by {n}: ${p} x {n} = ${r}. The total is ${r}."),
    ("A trip is {d} km and takes {h} hours. What is the average speed?", "Average speed is distance divided by time: {d} / {h} = {r} km per hour."),
    ("There are {n} people sharing {total} cookies equally. How many does each person get?", "Divide {total} by {n}: {total} / {n} = {r}. Each person gets {r} cookies."),
    ("A class has {n} students. {k} are absent. How many are present?", "Subtract the absent students: {n} - {k} = {r}. So {r} students are present."),
]
for template, answer_template in word_problems:
    for _ in range(12):
        if "average speed" in template:
            h = rng.choice([2, 3, 4, 5, 6])
            r = rng.choice([20, 30, 40, 50, 60, 80])
            d = h * r
            vals = {"d": d, "h": h, "r": r}
        elif "cookies" in template:
            n = rng.choice([2, 3, 4, 5, 6, 8, 10])
            r = rng.randint(2, 15)
            vals = {"n": n, "total": n * r, "r": r}
        elif "notebook" in template:
            p, n = rng.choice([2, 3, 4, 5, 6, 8, 10]), rng.randint(2, 8)
            vals = {"p": p, "n": n, "r": p * n}
        else:
            n = rng.randint(15, 80)
            k = rng.randint(1, n - 1)
            vals = {"n": n, "k": k, "r": n - k}
        add("word_problem", template.format(**vals), answer_template.format(**vals))

# Brief, factual explanations. Each core idea appears only a few times; there
# is no heavily repeated filler concept dominating the training stream.
facts = [
    ("the sky looks blue", "Air scatters shorter blue wavelengths of sunlight more strongly than red wavelengths, so scattered daylight looks blue."),
    ("the seasons", "Earth's axis is tilted. As Earth orbits the Sun, each hemisphere gets different amounts of direct sunlight and daylight."),
    ("photosynthesis", "Photosynthesis uses light energy to turn water and carbon dioxide into sugars. Plants release oxygen in the process."),
    ("gravity", "Gravity is the attraction associated with mass and energy. Near Earth, it pulls unsupported objects toward the ground."),
    ("an atom", "An atom is a basic unit of an element, with a nucleus of protons and usually neutrons, surrounded by electrons."),
    ("ice floating", "Ice floats because its crystal structure is less dense than liquid water."),
    ("the water cycle", "The water cycle moves water through evaporation, condensation, precipitation, and collection."),
    ("why the Moon shines", "The Moon looks bright because it reflects sunlight. It does not make its own visible light."),
    ("why we see lightning before thunder", "Light travels much faster than sound, so a lightning flash reaches you before its thunder."),
    ("a solar eclipse", "A solar eclipse happens when the Moon passes between Earth and the Sun and blocks some or all of the Sun from view."),
    ("a variable in programming", "A variable is a named place in a program for referring to a value."),
    ("a function in programming", "A function is reusable code that can take inputs, do work, and optionally return a result."),
    ("an algorithm", "An algorithm is a finite sequence of steps for solving a problem or completing a task."),
    ("HTTP", "HTTP is a protocol that web clients and servers use to exchange requests and responses."),
    ("machine learning", "Machine learning finds patterns in data so a model can make predictions or decisions on new inputs."),
    ("overfitting", "Overfitting is when a model learns training details too closely and performs poorly on new examples."),
    ("debugging", "Debugging means reproducing a problem, locating its cause, changing the code, and testing the fix."),
    ("a database", "A database stores organized information so it can be searched, changed, and retrieved."),
    ("an operating system", "An operating system manages a computer's hardware and provides services for programs."),
    ("a web browser", "A web browser requests web pages, interprets their code, and displays them for the user."),
    ("correlation", "Correlation describes how two variables change together. It does not by itself show that one causes the other."),
    ("a hypothesis", "A hypothesis is a proposed explanation or prediction that can be tested against evidence."),
    ("critical thinking", "Critical thinking means checking claims, assumptions, evidence, and alternatives before deciding what to believe."),
    ("evidence", "Evidence is information that supports or challenges a claim. Its quality and relevance matter."),
    ("a feedback loop", "A feedback loop occurs when a system's output influences what happens next in that system."),
    ("probability", "Probability describes how likely an event is, from impossible at zero to certain at one."),
    ("an average", "The arithmetic mean is found by adding the values and dividing by how many values there are."),
    ("a budget", "A budget is a plan for how income will be used for spending, saving, and other costs."),
    ("opportunity cost", "Opportunity cost is the value of the best alternative you give up when choosing something."),
    ("a metaphor", "A metaphor describes one thing as another to highlight a shared quality, without saying they are literally identical."),
    ("democracy", "Democracy is a system where political authority comes from the people, often through voting and representation."),
    ("inflation", "Inflation is a broad rise in prices over time, reducing what a unit of money can buy."),
    ("privacy", "Privacy is a person's ability to control access to information about themselves."),
    ("a trade-off", "A trade-off means gaining one benefit while accepting a cost or giving up another benefit."),
    ("a hypothesis test", "A hypothesis test compares data with a stated claim under assumptions. It does not prove the claim true."),
    ("a compiler", "A compiler translates source code into another form, often machine code, before the program runs."),
    ("an API", "An API is a defined way for one program to request services or data from another program."),
    ("encryption", "Encryption transforms readable data into a form that should be unreadable without the right key."),
    ("a neuron", "A neuron is a nerve cell that receives and sends signals in the nervous system."),
    ("DNA", "DNA is a molecule that stores genetic instructions used in the growth and function of living things."),
    ("a cell", "A cell is the smallest unit that can carry out the basic processes of life."),
    ("bacteria", "Bacteria are single-celled organisms. Some cause disease, while many are harmless or useful."),
    ("a virus", "A virus is an infectious agent that reproduces by entering host cells and using their machinery."),
    ("the Earth's rotation", "Earth rotates once roughly every 24 hours, producing the daily cycle of day and night."),
    ("the Earth's orbit", "Earth takes about one year to orbit the Sun."),
    ("sound", "Sound is a vibration that travels through a material such as air, water, or a solid."),
    ("a prime number", "A prime number is a whole number greater than one with exactly two positive divisors: one and itself."),
    ("a fraction", "A fraction represents part of a whole. Its numerator counts parts and its denominator gives the number of equal parts."),
    ("a main idea", "The main idea is the central point a passage makes; supporting details explain or illustrate it."),
    ("active listening", "Active listening means paying attention, checking your understanding, and responding to what the other person said."),
    ("a scientific theory", "A scientific theory is a well-supported explanation that accounts for evidence and makes testable predictions."),
    ("the greenhouse effect", "Greenhouse gases absorb and re-emit some outgoing heat, warming Earth's lower atmosphere."),
    ("renewable energy", "Renewable energy comes from sources that are naturally replenished, such as sunlight, wind, and flowing water."),
    ("a black hole", "A black hole is a region where gravity is so strong that nothing, not even light, can escape from within its event horizon."),
    ("the median", "The median is the middle value after numbers are sorted. For an even count, it is the mean of the two middle values."),
    ("a noun", "A noun names a person, place, thing, or idea."),
    ("a verb", "A verb expresses an action, event, or state of being."),
    ("a metaphor versus a simile", "A metaphor says one thing is another; a simile compares them using words such as 'like' or 'as'."),
]
facts.extend([
    ("evaporation", "Evaporation happens when liquid molecules gain enough energy to become gas."),
    ("condensation", "Condensation happens when a gas cools and changes into a liquid."),
    ("a habitat", "A habitat is the natural place where an organism lives and finds what it needs."),
    ("an ecosystem", "An ecosystem includes living organisms and the nonliving parts of their environment, interacting together."),
    ("a food chain", "A food chain shows how energy passes from one organism to another through feeding."),
    ("adaptation", "An adaptation is an inherited feature that can help an organism survive and reproduce in its environment."),
    ("a fossil", "A fossil is preserved evidence of ancient life, such as bones, shells, or traces in rock."),
    ("a magnet", "A magnet produces a magnetic field and attracts some materials, especially iron, nickel, and cobalt."),
    ("electric current", "Electric current is the flow of electric charge through a material or circuit."),
    ("energy", "Energy is a quantity that can be transferred or changed from one form to another; it is conserved in a closed system."),
    ("tides", "Tides are regular changes in sea level caused mainly by the Moon's gravity and, to a lesser extent, the Sun's gravity."),
    ("a map legend", "A map legend explains the symbols and colors used on the map."),
    ("a scale on a map", "Map scale compares a distance on the map with the matching distance in the real world."),
    ("a budget deficit", "A budget deficit occurs when spending is greater than income over a stated period."),
    ("a hypothesis and a theory", "A hypothesis is a testable proposal. A scientific theory is a broad explanation supported by substantial evidence."),
    ("a recession", "A recession is a significant decline in economic activity across an economy, though exact definitions vary by country."),
    ("a democracy and a republic", "A democracy gives people a role in political decisions. A republic has a head of state who is not a hereditary monarch; a country can be both."),
    ("the median and the mean", "The mean adds values and divides by their count. The median is the middle value after sorting them."),
    ("a right angle", "A right angle measures 90 degrees."),
    ("the perimeter of a rectangle", "The perimeter is the total distance around the rectangle: add all sides, or calculate 2 times the length plus 2 times the width."),
    ("the area of a rectangle", "The area of a rectangle is its length multiplied by its width."),
    ("a negative number", "A negative number is less than zero and is shown with a minus sign, such as -3."),
    ("a fraction in simplest form", "A fraction is in simplest form when its numerator and denominator have no common factor greater than one."),
    ("a renewable resource", "A renewable resource is replenished naturally on a timescale relevant to its use, such as sunlight or wind."),
    ("recycling", "Recycling processes used materials so they can become new products, reducing the need for some raw materials."),
    ("a password manager", "A password manager stores and helps create strong, distinct passwords. Protect it with a strong master password and available security features."),
    ("two-factor authentication", "Two-factor authentication requires a second proof of identity in addition to a password, such as an app code or security key."),
    ("a backup", "A backup is a separate copy of data that can be restored if the original is lost or damaged."),
    ("a software bug", "A software bug is an error or flaw that causes a program to behave in an unintended way."),
    ("open-source software", "Open-source software makes its source code available under a license that allows specified forms of use and modification."),
])
question_forms = ["what is {topic}?", "explain {topic} simply", "can you explain {topic}?"]
for topic, explanation in facts:
    # Vary the user wording, but keep each answer concept-specific. A repeated
    # generic opener such as "In simple terms" teaches the small model to emit
    # the same unrelated template for many different questions.
    for form in question_forms:
        add("knowledge", form.format(topic=topic), explanation)

add("knowledge", "what does a feedback loop mean?", "A feedback loop occurs when a system's output influences what happens next in that system.", weight=6.0)
add("knowledge", "what does a feedback loop do?", "A feedback loop uses a system's output to influence what happens next in that system.", weight=4.0)

# Logic demonstrations use different entities and premises so the model learns
# compact reasoning patterns instead of memorizing one repeated example.
logic_examples = [
    ("All robins are birds. Pip is a robin. Is Pip a bird?", "Yes. All robins are birds, and Pip is a robin, so Pip is a bird."),
    ("All squares are rectangles. Some rectangles are blue. Must every square be blue?", "No. The blue rectangles might not be squares. The premises do not say that every square is blue."),
    ("If it rains, the path gets wet. The path is wet. Did it definitely rain?", "No. Another cause, such as a hose, could make the path wet. The premises do not prove that it rained."),
    ("If the alarm is set, its light is on. The light is off. Is the alarm set?", "No, assuming the rule has no exceptions. If the alarm were set, the light would be on."),
    ("Ana is older than Ben, and Ben is older than Kai. Who is oldest?", "Ana is oldest: she is older than Ben, who is older than Kai."),
    ("A box is left of a lamp. The lamp is left of a chair. Where is the box?", "The box is left of the chair, because both stated positions are ordered from left to right."),
    ("There are three red balls and two blue balls. Are most balls red?", "Yes. Three of the five balls are red, which is more than half."),
    ("No poets are robots. Lee is a poet. Can Lee be a robot?", "No. The premise excludes poets from the group of robots, and Lee is a poet."),
    ("If a number is divisible by 4, it is even. 12 is divisible by 4. Is 12 even?", "Yes. The rule says numbers divisible by 4 are even, and 12 is divisible by 4."),
    ("All mammals breathe air. A dolphin is a mammal. Does a dolphin breathe air?", "Yes. By the stated rule, a dolphin breathes air because it is a mammal."),
    ("Every book on the desk is blue. This book is on the desk. What color is it?", "It is blue, according to the rule and the fact that it is on the desk."),
    ("Some students play chess. Does that mean all students play chess?", "No. 'Some' means at least one, not all students."),
    ("If I study, I pass. I did not study. Can we conclude I failed?", "No. The rule does not say studying is the only way to pass."),
    ("A is taller than B. B is taller than C. Is C taller than A?", "No. The comparisons imply A is taller than C."),
    ("There are 4 green cards and 4 yellow cards. Are green cards the majority?", "No. There are four of each, so neither color is a majority."),
    ("All fruits in this bowl are ripe. A pear is in the bowl. Is the pear ripe?", "Yes, if the pear is one of the fruits in the bowl, the rule says it is ripe."),
]
for prompt, answer in logic_examples:
    add("logic", prompt, answer)

# Create more distinct transitive-order questions with changing names and facts.
names = ["Ava", "Bo", "Cam", "Dee", "Eli", "Fern", "Gus", "Hal", "Ira", "Jo"]
for i in range(24):
    a, b, c = rng.sample(names, 3)
    relation = rng.choice(["older than", "faster than", "taller than", "earlier than"])
    prompt = f"{a} is {relation} {b}, and {b} is {relation} {c}. Who is first in that order?"
    answer = f"{a} is first: {a} is {relation} {b}, who is {relation} {c}."
    add("logic", prompt, answer)

conditional_examples = [
    ("If a file is saved, it appears in the folder. The file is not in the folder. Was it saved?", "Under the stated rule, no. If it had been saved, it would appear in the folder."),
    ("If the switch is on, the light is on. The light is on. Must the switch be on?", "No. Another source could light the bulb, so the rule does not prove the switch is on."),
    ("If a shape is a square, it has four sides. This shape has four sides. Is it a square?", "Not necessarily. Other shapes, such as rectangles, also have four sides."),
    ("If a plant gets no water, it wilts. The plant has water. Must it be healthy?", "No. Water alone does not guarantee that the plant is healthy; other conditions matter."),
    ("If a bus is late, I miss the meeting. The bus was on time. Did I attend?", "The rule does not tell us whether you attended. Other events could still prevent you from going."),
    ("If a number is even, it is divisible by 2. 18 is even. Is it divisible by 2?", "Yes. The given rule says every even number is divisible by 2."),
    ("If a file is encrypted, it needs a key to read. This file needs a key. Is it definitely encrypted?", "Not necessarily. The rule gives one reason a key may be needed, but there could be other reasons."),
    ("If it is a weekday, the office is open. It is Saturday. Is the office open?", "The rule only says what happens on weekdays. It does not tell us the office's Saturday hours."),
    ("All bicycles have wheels. This vehicle has wheels. Is it definitely a bicycle?", "No. Cars and many other vehicles have wheels too."),
    ("All squares have four equal sides. This shape has four equal sides. Must it be a square?", "No. A rhombus can have four equal sides without having four right angles."),
    ("Some birds cannot fly. Does that mean no birds can fly?", "No. 'Some' means at least a few; it does not describe every bird."),
    ("No reptiles are mammals. A snake is a reptile. Is it a mammal?", "No. The premises place reptiles and mammals in separate groups, and a snake is a reptile."),
    ("All the coins in the jar are silver. One coin is gold. Is it in the jar?", "No, not if the rule is true and the coin is gold, because every coin in the jar is silver."),
    ("If Mia studies, she finishes the quiz. Mia finished the quiz. Did she study?", "We cannot know. She might have finished it another way; the rule does not say studying is the only route."),
    ("If a number ends in zero, it is divisible by ten. 70 ends in zero. Is it divisible by ten?", "Yes. The stated rule applies to 70, so it is divisible by ten."),
    ("A is to the left of B. B is to the left of C. Is C to the left of A?", "No. The order is A, then B, then C from left to right."),
    ("A red box is heavier than a blue box. The blue box is heavier than a green box. Which is lightest?", "The green box is lightest because the blue box is heavier than it, and the red box is heavier than the blue box."),
    ("A recipe needs 2 eggs for one cake. I have 4 eggs. How many cakes can I make?", "At most 2 cakes, if eggs are the only limit: 4 divided by 2 is 2."),
    ("A train leaves at 3 and arrives at 5. How long is the trip?", "The trip lasts 2 hours, assuming both times use the same time zone and day."),
    ("I have 3 red pens and 3 blue pens. Are there more red pens?", "No. There are three of each, so the counts are equal."),
    ("Every student passed the quiz. Lee is a student. Did Lee pass?", "Yes, if the statement is true and Lee is a student, Lee passed the quiz."),
    ("No apples are oranges. This fruit is an apple. Can it also be an orange?", "No, not under the stated rule. Apples and oranges are treated as separate groups here."),
    ("The alarm rings only when smoke is detected. It rang. Was smoke detected?", "Yes, if 'only when' is used literally: ringing requires smoke detection."),
    ("The alarm rings if smoke is detected. It rang. Was smoke detected?", "Not necessarily. The wording says smoke can make it ring, but another cause may also make it ring."),
]
for prompt, answer in conditional_examples:
    add("logic", prompt, answer)

# Practical planning and decision help; each topic appears in a few distinct
# phrasings, not dozens of near-identical filler records.
plans = [
    ("study for an exam", "list the topics, find weak areas, use short retrieval-practice sessions, and finish with a timed practice test"),
    ("start exercising", "choose a realistic weekly schedule, begin gently, track consistency, and increase gradually"),
    ("save money", "review recent spending, set a clear target, automate a small transfer, and check progress weekly"),
    ("clean my room", "collect rubbish, group laundry, clear one surface, then return items to their homes"),
    ("learn programming", "choose one language, practice fundamentals, build tiny projects, and learn from errors"),
    ("prepare a presentation", "choose one takeaway, make a short outline, use relevant evidence, and rehearse aloud"),
    ("write an essay", "clarify the question, form a thesis, outline claims and evidence, draft, then revise for clarity"),
    ("plan a trip", "set dates and a budget, confirm transport and lodging, group activities by area, and leave flexible time"),
    ("organize my week", "write down commitments, choose a few priorities, block time for them, and leave room for changes"),
    ("get better at drawing", "practice basic shapes, draw from observation, review one thing to improve, and repeat regularly"),
    ("prepare for a difficult conversation", "decide what outcome you want, use specific examples, listen carefully, and keep your tone respectful"),
    ("build a website", "start with one page, sketch its content, build the simplest version, and test it on a phone"),
    ("read more", "choose a book you actually want to read, set a small daily target, and keep it somewhere visible"),
    ("reduce screen time", "notice when you reach for your phone, turn off nonessential alerts, and choose a replacement activity"),
    ("start a garden", "check the light and climate, choose easy plants, prepare the soil, and water consistently"),
    ("practice a language", "listen to short material, learn useful phrases, speak regularly, and review mistakes"),
    ("manage a busy inbox", "unsubscribe from unwanted mail, sort messages by action needed, and schedule two short times to check it"),
    ("prepare for a long trip", "confirm bookings, pack essentials, check the route, and keep important details available offline"),
    ("build a morning routine", "choose one or two useful actions, prepare what you need the night before, and keep the routine realistic"),
    ("learn basic photography", "practice light and composition, take several versions of one scene, and compare what works"),
    ("organize a group project", "agree on the goal, divide work into clear owners, set check-in dates, and track decisions"),
    ("prepare for a doctor's appointment", "write down symptoms, when they began, medicines you take, and questions you want answered"),
    ("reduce clutter", "choose one small area, sort items into keep, donate, and discard, then finish before moving on"),
    ("create a study schedule", "list deadlines, estimate how long each task takes, schedule short sessions, and review the plan weekly"),
    ("start a small business", "identify a customer problem, test a simple offer, estimate costs, and learn from early feedback"),
    ("plan a family meal", "check dietary needs, choose a simple menu, make a shopping list, and prepare anything that can be done early"),
    ("learn to play guitar", "practice a few basic chords, change between them slowly, and use a short song to build consistency"),
    ("prepare for a job search", "update your resume, identify suitable roles, tailor each application, and keep track of follow-ups"),
    ("make a room more comfortable", "notice light, temperature, noise, and seating, then change one thing that bothers you most"),
    ("get ready for a move", "confirm dates, sort belongings, label boxes by room, and keep important documents and essentials accessible"),
    ("improve my note taking", "write key ideas in your own words, mark questions, and review the notes soon after class"),
    ("set a realistic goal", "describe a measurable outcome, choose a time frame, identify a first action, and plan how to check progress"),
    ("learn to cook more meals", "choose a few simple recipes, repeat them until comfortable, and keep a list of ingredients you use often"),
    ("prepare for a test", "check the topics, practice retrieval without notes, review mistakes, and get enough rest before the test"),
    ("make a presentation more engaging", "open with a clear question or example, keep slides uncluttered, and connect each section to your main point"),
    ("review a monthly budget", "compare actual spending with the plan, note one surprise, adjust the next month, and keep a small buffer"),
]
for goal, plan in plans:
    add("planning", f"help me {goal}", f"Start with a clear, realistic outcome. Then {plan}. Put the first small action on your calendar.")
    add("planning", f"how can i make progress on {goal}?", f"Break it into a small routine: {plan}. Start with one part today and adjust after you try it.")

# Writing assistance with examples that train transformations and concise style.
writing_examples = [
    ("make this polite: send me the report now", "Could you please send me the report when you have a moment? Thank you."),
    ("write a short apology for being late", "I'm sorry I was late. I should have planned better, and I appreciate your patience."),
    ("write a friendly meeting reminder", "Hi! Just a friendly reminder that we're meeting tomorrow at 10 a.m. Please let me know if anything changes."),
    ("improve this: the test was good and showed good results", "The test was well designed and produced encouraging results."),
    ("write a concise thank-you note", "Thank you for your time and help. I truly appreciate it."),
    ("make this less blunt: i need this today", "Could you send this over today if possible? I would really appreciate it."),
    ("write a subject line for a project update", "Project update: progress, next steps, and open questions"),
    ("write a short invitation to lunch", "Would you like to get lunch with me this week? Let me know what day works for you."),
    ("make this clearer: due to the fact that we were late, we missed it", "Because we were late, we missed it."),
    ("write a polite decline", "Thank you for thinking of me. I cannot commit to this right now, but I appreciate the invitation."),
    ("write a brief congratulations message", "Congratulations on the great news! I hope you get a chance to celebrate."),
    ("rewrite this more directly: it is possible that we may need to change the plan", "We may need to change the plan."),
    ("write a professional follow-up email opener", "Hello, I am following up on my earlier message about the project. Do you have an update?"),
    ("how do i make writing clearer?", "Use specific words, keep one main idea per paragraph, remove repetition, and revise for the reader."),
    ("write a one-sentence description of a quiet beach", "The quiet beach was washed in soft evening light, with only the tide moving along the shore."),
    ("write a short note asking to reschedule", "Could we reschedule our meeting? I am sorry for the change and can work around your availability."),
    ("make this sound appreciative: you helped me", "I really appreciate your help. It made a difference, and I am grateful you took the time."),
    ("write a simple reminder to bring lunch", "Reminder: please bring your lunch tomorrow."),
    ("write a short welcome message for a new teammate", "Welcome to the team! We are glad you are here. Let us know how we can help you get settled."),
    ("turn this into a question: you can send the file today", "Could you send the file today?"),
    ("write a brief message saying i will be five minutes late", "I am running about five minutes late. Sorry for the delay; I will be there soon."),
    ("make this sentence more specific: the project went well", "The project met its deadline, stayed within budget, and received positive feedback from the client."),
    ("write a short thank-you for an interview", "Thank you for meeting with me today. I enjoyed learning about the role and the team's work."),
    ("write a kind message to someone having a hard week", "I am sorry this week has been difficult. I am thinking of you, and I am here if you want to talk."),
    ("write a brief request for a deadline extension", "Would it be possible to extend the deadline to Friday? I can send the completed work by then. Thank you for considering it."),
    ("make this more concise: at this point in time we are unable to proceed", "We cannot proceed right now."),
    ("write a short message thanking a neighbor", "Thank you for bringing in my package. I really appreciate your help."),
    ("write a reminder about a team deadline", "Reminder: our team draft is due Thursday. Please add your section before the end of the day."),
    ("rewrite this in a warmer tone: your request was received", "Thanks for sending your request. I have received it and will follow up soon."),
    ("write a short note to welcome a guest", "Welcome! We are happy to have you here. Please let us know if you need anything."),
    ("write a message asking for feedback", "When you have time, could you share feedback on the draft? I would especially value your thoughts on the structure."),
    ("make this more professional: i can't come today", "I am unable to attend today. I apologize for the short notice."),
    ("write a short note congratulating a graduate", "Congratulations on graduating! Your hard work paid off, and I hope you enjoy celebrating this achievement."),
    ("improve this sentence: we did the work fast", "We completed the work ahead of schedule."),
    ("write a short message to check on a friend", "Hi, I have been thinking about you. How are you doing? No pressure to reply right away."),
    ("write a polite request for the meeting notes", "Could you please share the meeting notes when they are ready? Thank you."),
    ("make this clearer: it was decided to move the meeting", "We decided to move the meeting."),
    ("write a short description of a rainy afternoon", "Rain tapped against the window while the street blurred into silver reflections."),
    ("write a brief reminder to submit a form", "Reminder: please submit the form by Monday. Let me know if you need help."),
    ("write a concise project status update", "The first draft is complete. I am reviewing the data now and expect to share the revision on Friday."),
    ("write a kind message after someone loses a pet", "I am so sorry about your pet. They were clearly loved, and I am here if you want to talk about them."),
    ("write a short note declining a meeting", "Thank you for the invitation. I cannot attend this time, but please send any notes or next steps."),
    ("make this sentence more direct: we would like to request that you reply", "Please reply when you can."),
    ("write a quick birthday message", "Happy birthday! I hope your day brings good company, a little fun, and something you love."),
    ("write a brief message asking someone to call", "Could you give me a call when you have a moment? Thanks."),
    ("rewrite this to sound less certain: this will definitely fix it", "This may fix the issue, but we should test it to be sure."),
    ("write a short introduction for a report", "This report summarizes the main findings, explains the limits of the analysis, and recommends next steps."),
    ("write a friendly message thanking a colleague", "Thanks for stepping in to help with the presentation. Your input made the final version much stronger."),
    ("make this sentence simpler: the implementation was unsuccessful", "The change did not work."),
    ("write a brief note asking for directions", "Hi, could you tell me the best way to get to the library from here?"),
    ("write a short note to reschedule an appointment", "I need to reschedule my appointment. Could you let me know what times are available next week?"),
    ("write a short message to encourage a teammate", "You have made solid progress. Keep going, and let me know if there is anything I can do to help."),
    ("make this less vague: we need to improve the service", "We should reduce support wait times and resolve more customer issues on the first call."),
    ("write a short thank-you after a helpful conversation", "Thank you for taking the time to talk with me. Your perspective helped me think about the problem differently."),
    ("write a concise out-of-office message", "I am away until Tuesday and will reply when I return. For urgent issues, please contact the team inbox."),
    ("write a simple note inviting a friend for a walk", "Would you like to go for a walk with me this weekend? We can choose an easy route."),
    ("write a short message to confirm a booking", "Hello, I am writing to confirm my booking for Saturday at 2 p.m. Please let me know if you need any details."),
    ("improve this sentence: the instructions were not easy to understand", "The instructions were unclear."),
    ("write a brief apology for missing a call", "Sorry I missed your call. I was unavailable at the time. Please let me know when it would be convenient to talk."),
    ("write a short message asking a group to choose a time", "Which of these times works best for everyone: Tuesday afternoon or Wednesday morning?"),
    ("make this more polite: fix the typo", "Could you please correct the typo when you have a chance?"),
    ("write a short review praising good service", "The staff were helpful, the service was prompt, and the whole experience was easy."),
    ("write a short note saying i received the document", "I received the document. Thank you for sending it over."),
]
for prompt, answer in writing_examples:
    add("writing", prompt, answer)

# Simple choices: ask about priorities and uncertainty instead of pretending
# there is one universally correct answer.
decisions = [
    ("buy a cheaper laptop now", "save for a stronger laptop", "your current needs, total cost, expected lifetime, repairability, and whether waiting blocks important work"),
    ("take the faster route", "take the scenic route", "arrival time, safety, stress, cost, and how much you value the experience"),
    ("work tonight", "rest tonight", "deadline risk, current energy, task importance, and whether rest would improve tomorrow's work"),
    ("study alone", "join a study group", "your focus, the group's quality, accountability, and whether discussion helps this subject"),
    ("rent near work", "commute from farther away", "rent, travel time, transport reliability, and how much you value living space"),
    ("repair my old phone", "buy a new phone", "repair cost, remaining useful life, security updates, and your budget"),
    ("accept a new responsibility", "keep my current workload", "your priorities, time, support, and what you would need to stop doing"),
    ("cook at home", "order dinner", "time, cost, energy, ingredients, and what would make tonight easier"),
    ("send the message now", "wait until tomorrow", "urgency, your emotional state, and whether a short review would improve the message"),
    ("take a familiar class", "try a challenging class", "your goals, workload, support, interest, and the cost of changing later"),
    ("move to a new city", "stay where i am", "work, relationships, living costs, support, and what you hope the move will change"),
    ("buy used equipment", "buy new equipment", "price, condition, warranty, reliability, and how often you will use it"),
    ("take a stable job", "try a riskier opportunity", "your financial needs, support network, risk tolerance, learning goals, and whether you can return to a stable path"),
    ("study in the morning", "study in the evening", "your energy, schedule, interruptions, and which time helps you remember more"),
    ("rent a smaller place alone", "share a larger place", "privacy, cost, location, compatibility with housemates, and how much space you need"),
    ("buy a car", "use public transport", "upfront and ongoing costs, travel needs, reliability, safety, and the routes available to you"),
    ("finish a course", "change to a different course", "your goals, interest, remaining time, cost, and whether the current course is still useful"),
    ("send a detailed explanation", "send a short reply", "the reader's needs, urgency, complexity, and how much context they already have"),
    ("work on the hardest task first", "start with an easier task", "your energy, deadlines, momentum, and the consequences of delaying each task"),
    ("travel alone", "travel with a group", "safety, cost, independence, shared interests, and the kind of trip you want"),
]
for first, second, criteria in decisions:
    answer = f"Compare them using {criteria}. Rank those factors by importance, check your biggest uncertainty, then choose the better overall trade-off."
    add("decision", f"should i {first} or {second}?", answer)
    add("decision", f"help me choose between {first} and {second}", answer)
    add("decision", f"which option is better: {first} or {second}?", answer)

safety = [
    ("i have severe chest pain. what should i do?", "Severe chest pain can be an emergency. Call your local emergency number now and follow the dispatcher's instructions."),
    ("can you diagnose my symptoms?", "I can share general information, but I cannot diagnose you. A qualified clinician can assess your symptoms and history."),
    ("i want to hurt myself", "I'm sorry you are in so much pain. Contact emergency services or a crisis line now, and reach out to someone you trust who can stay with you."),
    ("should i trust a confident answer?", "Confidence is not evidence. Check the reasoning, assumptions, and reliable sources before relying on an answer."),
    ("is one example enough to prove a general rule?", "One counterexample can disprove a universal rule, but one supporting example is usually not enough to prove it."),
    ("can you give me current legal advice?", "I can explain general concepts, but laws depend on place and time. For advice on your situation, consult a qualified local lawyer."),
    ("is this investment guaranteed to make money?", "No investment return is guaranteed. Consider the risks, fees, time horizon, and whether you can afford a loss."),
    ("i may have taken too much medicine", "Contact your local poison center or emergency service now. Tell them the medicine, amount, and time taken, and follow their instructions."),
]
for prompt, answer in safety:
    add("safety", prompt, answer)

# Shuffle deterministically so minibatches mix task types. Duplicate prompts
# are rejected above, and core concepts use only a small set of varied examples.
rng.shuffle(rows)
counts = Counter(row["category"] for row in rows)
assert len(rows) >= 1000, f"unexpectedly small corpus: {len(rows)}"
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
print(f"wrote {len(rows)} unique examples to {OUT}")
print("categories: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
