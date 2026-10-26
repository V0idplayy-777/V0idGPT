#!/usr/bin/env python3
"""
Build the v0idgpt-4-135M conversational dataset.

Inputs
  scripts/data/*.txt            hand-written conversations  (  "#category" then "U:"/"A:" lines )
Outputs
  models/v0idgpt-4-135M/dataset.jsonl   {"category":..., "messages":[{"role","content"},...]}
  models/v0idgpt-4-135M/dataset.txt     human-readable transcript (this is the "10,000+ lines")
  models/v0idgpt-4-135M/dataset-stats.json

Everything numeric is computed, never written by hand, so the arithmetic in the corpus is exact.
Deterministic: fixed seed, no network, no external data.
"""
import json, os, random, re, sys

SEED = 20251025
RNG = random.Random(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "models", "v0idgpt-4-135M")

CONVS = []          # list of {"category":str, "messages":[...]}

# ----------------------------------------------------------------------------- helpers
def pick(seq):
    return RNG.choice(seq)

def maybe(p=0.5):
    return RNG.random() < p

def C(cat, *turns):
    """turns: pairs of user/assistant strings, starting with user."""
    msgs = []
    for i, t in enumerate(turns):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t.strip()})
    assert len(msgs) % 2 == 0 and msgs[-1]["role"] == "assistant", msgs
    CONVS.append({"category": cat, "messages": msgs})

def num(n):
    """Natural number formatting: 1234 -> '1,234' sometimes."""
    if abs(n) >= 1000 and maybe(0.35):
        return f"{n:,}"
    return str(n)

def money(x):
    if float(x).is_integer():
        return f"${int(x)}"
    return f"${x:.2f}"

# ----------------------------------------------------------------------------- math helpers
def steps_add(a, b):
    """Short, correct explanation for a multi-digit addition."""
    if b == 0:
        return f"{a} + 0 is just {a}."
    if a % 10 + b % 10 >= 10:
        ta = (a // 10) * 10
        tb = (b // 10) * 10
        return (f"{a} + {b} = {a + b}. Easiest way: {a} + {tb} = {a + tb}, "
                f"then add the {b % 10} = {a + b}.")
    return f"{a} + {b} = {a + b}."

def steps_sub(a, b):
    if b > a:
        return f"{a} - {b} = -{b - a}. The result goes negative."
    if a % 10 < b % 10:
        return (f"{a} - {b} = {a - b}. Borrow one ten: {a} - {a % 10} = {a - a % 10}, "
                f"then take off the remaining {b - a % 10}.")
    return f"{a} - {b} = {a - b}."

WORD_NOUNS = [
    ("apple", "apples"), ("pencil", "pencils"), ("book", "books"), ("marble", "marbles"),
    ("sticker", "stickers"), ("egg", "eggs"), ("cookie", "cookies"), ("orange", "oranges"),
    ("token", "tokens"), ("card", "cards"), ("ticket", "tickets"), ("coin", "coins"),
    ("candle", "candles"), ("bottle", "bottles"), ("cake", "cakes"), ("balloon", "balloons"),
]
NAMES = ["sam", "alex", "mia", "jordan", "noah", "priya", "leo", "ava", "kai", "zoe",
         "maya", "finn", "ivy", "omar", "ruby", "theo", "nina", "luca", "erin", "seth"]

def plural(n, sing, plur):
    return f"{num(n)} {sing if n == 1 else plur}"

# ----------------------------------------------------------------------------- 1. arithmetic
def build_arithmetic(n_target):
    made = 0
    op_mix = [("add", 0.30), ("sub", 0.24), ("mul", 0.28), ("div", 0.12), ("mix", 0.06)]

    def choose_op():
        r = RNG.random()
        acc = 0
        for op, w in op_mix:
            acc += w
            if r <= acc:
                return op
        return "add"

    while made < n_target:
        op = choose_op()
        if op == "add":
            if maybe(0.55):
                a, b = RNG.randint(2, 99), RNG.randint(2, 99)
            else:
                a, b = RNG.randint(100, 999), RNG.randint(10, 999)
            r = a + b
            q = pick([
                f"what's {a} + {b}", f"what is {a} plus {b}", f"{a} + {b} = ?",
                f"can you add {a} and {b}", f"compute {a} + {b}", f"{a} plus {b}?",
                f"what's {a} plus {b} equal", f"quick one: {a} + {b}",
                f"{a}+{b}", f"add {a} to {b}", f"what does {a} + {b} make",
            ])
            forms = [f"{r}.", f"{a} + {b} = {r}.", f"That's {r}.", f"It's {r}."]
            if (a >= 100 or b >= 100) and maybe(0.30):
                forms.append(steps_add(a, b))
            a_ = pick(forms)
        elif op == "sub":
            a = RNG.randint(10, 999)
            b = RNG.randint(1, a)
            r = a - b
            q = pick([
                f"what's {a} - {b}", f"what is {a} minus {b}", f"{a} - {b} = ?",
                f"subtract {b} from {a}", f"compute {a} - {b}", f"{a} take away {b}",
                f"what's {a} minus {b}", f"{a}-{b}?", f"how much is {a} - {b}",
            ])
            forms = [f"{r}.", f"{a} - {b} = {r}.", f"That's {r}."]
            if (a >= 100 or b >= 40) and maybe(0.30):
                forms.append(steps_sub(a, b))
            a_ = pick(forms)
        elif op == "mul":
            if maybe(0.6):
                a, b = RNG.randint(2, 12), RNG.randint(2, 12)
            else:
                a, b = RNG.randint(11, 25), RNG.randint(2, 9)
            r = a * b
            q = pick([
                f"what's {a} times {b}", f"what is {a} x {b}", f"{a} * {b}",
                f"{a} times {b}?", f"compute {a} multiplied by {b}", f"what's {a} × {b}",
                f"multiply {a} by {b}", f"{a}x{b} = ?", f"how much is {a} times {b}",
            ])
            forms = [f"{r}.", f"{a} × {b} = {r}.", f"That's {r}.", f"{r}."]
            if a > 10 and maybe(0.4):
                forms.append(f"{a} × {b} = {r}. Break it up: {a} × {b - 1} = {a * (b - 1)}, "
                             f"plus one more {a} = {r}.")
            a_ = pick(forms)
        elif op == "div":
            b = RNG.randint(2, 12)
            r = RNG.randint(2, 15)
            a = b * r
            if maybe(0.25):
                q = pick([f"what's {a} divided by {b}", f"{a} / {b}", f"divide {a} by {b}"])
                a_ = pick([f"{r}.", f"{a} ÷ {b} = {r}.", f"That's {r}.", f"{r} — {b} goes into {a} {r} times."])
            else:
                q = pick([f"what's {a} divided by {b}", f"{a} / {b}", f"divide {a} by {b}",
                          f"how many times does {b} go into {a}", f"{a} ÷ {b} = ?"])
                a_ = pick([f"{r}.", f"{a} ÷ {b} = {r}.", f"{r} times.", f"{r}."])
        else:  # order of operations
            mode = RNG.randint(0, 2)
            if mode == 0:
                a, b, c = RNG.randint(2, 12), RNG.randint(2, 9), RNG.randint(2, 9)
                q = f"what's {a} + {b} * {c}"
                a_ = f"{a + b * c}. Multiply first: {b} × {c} = {b * c}, then add {a}."
            elif mode == 1:
                a, b, c = RNG.randint(2, 12), RNG.randint(2, 6), RNG.randint(2, 6)
                q = f"what's ({a} + {b}) * {c}"
                a_ = f"{(a + b) * c}. Brackets first: {a} + {b} = {a + b}, then × {c} = {(a + b) * c}."
            else:
                a, b, c = RNG.randint(10, 60), RNG.randint(2, 9), RNG.randint(2, 9)
                q = f"what's {a} - {b} * {c}"
                a_ = f"{a - b * c}. Multiply first: {b} × {c} = {b * c}, then {a} - {b * c} = {a - b * c}."
        # optional second turn: "how'd you get that" / "nice"
        if maybe(0.14):
            follow = pick(["how'd you get that", "why", "show your work", "explain", "how"])
            expl = {
                "add": f"Straight addition. {a} + {b}: add the tens, then the units — {a + b}.",
                "sub": f"Take {b} off {a} in pieces: {a} - {b} = {a - b}.",
                "mul": f"{a} × {b} means {b} groups of {a}, which totals {a * b}.",
                "div": f"Division undoes multiplication: {b} × {r} = {a}, so {a} ÷ {b} = {r}.",
            }[op if op != "mix" else "mul"]
            C("math", q, a_, follow, expl)
        else:
            C("math", q, a_)
        made += 1

def build_word_math(n_target):
    made = 0
    while made < n_target:
        kind = RNG.randint(0, 7)
        name = pick(NAMES).capitalize()
        sing, plur = pick(WORD_NOUNS)
        if kind == 0:
            a, b = RNG.randint(2, 30), RNG.randint(2, 30)
            q = pick([f"i have {a} {plur} and buy {b} more, how many do i have",
                      f"{name} has {a} {plur} and gets {b} more. how many now?",
                      f"if i start with {a} {plur} and add {b}, what's the total"])
            C("math", q, f"{plural(a + b, sing, plur)}.")
        elif kind == 1:
            a = RNG.randint(10, 40)
            b = RNG.randint(1, a - 1)
            q = pick([f"a box has {a} {plur}, i take out {b}. how many are left",
                      f"there are {a} {plur} and {b} get used. how many remain",
                      f"{a} {plur} minus {b} {plur} is how many"])
            C("math", q, f"{plural(a - b, sing, plur)}.")
        elif kind == 2:
            price = RNG.randint(2, 20)
            n = RNG.randint(3, 12)
            q = pick([f"{n} {plur} cost {money(price)} each, what's the total",
                      f"if one {sing} is {money(price)}, how much do {n} cost",
                      f"i buy {n} {plur} at {money(price)} each. total?"])
            C("math", q, f"{money(price * n)}. {n} × {money(price)} = {money(price * n)}.")
        elif kind == 3:
            pct = pick([10, 20, 25, 50, 5, 75])
            base = pick([20, 40, 60, 80, 120, 200])
            off = base * pct // 100
            q = pick([f"a {money(base)} item is {pct}% off. what's the new price",
                      f"what's {pct}% off {money(base)}",
                      f"{pct}% discount on {money(base)}, final price?"])
            C("math", q, f"{money(base - off)}. {pct}% of {money(base)} is {money(off)}, "
                         f"so {money(base)} - {money(off)} = {money(base - off)}.")
        elif kind == 4:
            per = RNG.randint(2, 9)
            total = per * RNG.randint(3, 12)
            q = pick([f"{total} {plur} split between {per} people. how many each",
                      f"divide {total} {plur} equally into {per} groups",
                      f"i have {total} {plur} and {per} friends, how do we split them evenly"])
            C("math", q, f"{plural(total // per, sing, plur)} each.")
        elif kind == 5:
            speed = pick([20, 30, 40, 50, 60])
            hours = RNG.randint(2, 6)
            q = pick([f"a car goes {speed} km/h for {hours} hours. how far does it go",
                      f"if i drive {speed} km/h for {hours} hours, what distance do i cover",
                      f"distance covered at {speed} km/h for {hours} h?"])
            C("math", q, f"{num(speed * hours)} km. {speed} × {hours} = {num(speed * hours)}.")
        elif kind == 6:
            weeks = RNG.randint(2, 12)
            q = pick([f"how many days in {weeks} weeks", f"{weeks} weeks is how many days",
                      f"convert {weeks} weeks to days"])
            C("math", q, f"{num(weeks * 7)} days.")
        else:
            a, b = RNG.randint(2, 12), RNG.randint(2, 12)
            q = pick([f"a rectangle is {a} by {b}. what's the area",
                      f"area of a {a}x{b} rectangle",
                      f"i have a garden {a} metres by {b} metres, how much area is that"])
            C("math", q, f"{num(a * b)} square units. {a} × {b} = {num(a * b)}.")
        made += 1

# ----------------------------------------------------------------------------- 2. knowledge
CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Australia", "Canberra"), ("Canada", "Ottawa"),
    ("Brazil", "Brasília"), ("Kenya", "Nairobi"), ("Egypt", "Cairo"), ("Norway", "Oslo"),
    ("Spain", "Madrid"), ("Italy", "Rome"), ("Germany", "Berlin"), ("India", "New Delhi"),
    ("Mexico", "Mexico City"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
    ("Portugal", "Lisbon"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
    ("Poland", "Warsaw"), ("Turkey", "Ankara"), ("South Korea", "Seoul"),
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"),
    ("Morocco", "Rabat"), ("Nigeria", "Abuja"), ("Peru", "Lima"), ("Colombia", "Bogotá"),
    ("Greece", "Athens"), ("Ireland", "Dublin"), ("Netherlands", "Amsterdam"),
    ("Belgium", "Brussels"), ("Austria", "Vienna"), ("Switzerland", "Bern"),
    ("Denmark", "Copenhagen"), ("Iceland", "Reykjavík"), ("New Zealand", "Wellington"),
    ("Saudi Arabia", "Riyadh"), ("Philippines", "Manila"), ("Malaysia", "Kuala Lumpur"),
    ("Ukraine", "Kyiv"), ("Russia", "Moscow"), ("China", "Beijing"),
    ("Mongolia", "Ulaanbaatar"), ("Nepal", "Kathmandu"), ("Bangladesh", "Dhaka"),
    ("Pakistan", "Islamabad"), ("Cuba", "Havana"), ("Jamaica", "Kingston"),
    ("Ethiopia", "Addis Ababa"), ("Ghana", "Accra"), ("Senegal", "Dakar"),
    ("Tanzania", "Dodoma"), ("Uganda", "Kampala"), ("Zimbabwe", "Harare"), ("Qatar", "Doha"),
]
ELEMENTS = [
    ("H", "hydrogen"), ("He", "helium"), ("Li", "lithium"), ("C", "carbon"),
    ("N", "nitrogen"), ("O", "oxygen"), ("Na", "sodium"), ("Mg", "magnesium"),
    ("Al", "aluminium"), ("Si", "silicon"), ("P", "phosphorus"), ("S", "sulfur"),
    ("Cl", "chlorine"), ("K", "potassium"), ("Ca", "calcium"), ("Fe", "iron"),
    ("Cu", "copper"), ("Zn", "zinc"), ("Ag", "silver"), ("Sn", "tin"), ("I", "iodine"),
    ("Au", "gold"), ("Hg", "mercury"), ("Pb", "lead"), ("Ne", "neon"), ("Ar", "argon"),
    ("Ni", "nickel"), ("Pt", "platinum"), ("W", "tungsten"), ("Ti", "titanium"),
]
BOOKS = [
    ("1984", "George Orwell"), ("Animal Farm", "George Orwell"),
    ("Pride and Prejudice", "Jane Austen"), ("Moby-Dick", "Herman Melville"),
    ("The Hobbit", "J. R. R. Tolkien"), ("The Lord of the Rings", "J. R. R. Tolkien"),
    ("Harry Potter and the Philosopher's Stone", "J. K. Rowling"),
    ("Dune", "Frank Herbert"), ("Frankenstein", "Mary Shelley"), ("Dracula", "Bram Stoker"),
    ("The Great Gatsby", "F. Scott Fitzgerald"), ("To Kill a Mockingbird", "Harper Lee"),
    ("Crime and Punishment", "Fyodor Dostoevsky"), ("War and Peace", "Leo Tolstoy"),
    ("The Catcher in the Rye", "J. D. Salinger"), ("Brave New World", "Aldous Huxley"),
    ("The Old Man and the Sea", "Ernest Hemingway"), ("Beloved", "Toni Morrison"),
    ("Things Fall Apart", "Chinua Achebe"),
    ("One Hundred Years of Solitude", "Gabriel García Márquez"),
    ("Neuromancer", "William Gibson"), ("Fahrenheit 451", "Ray Bradbury"),
]
SCIENCE = [
    ("What temperature does water boil at?", "100°C at sea level (212°F)."),
    ("What temperature does water freeze at?", "0°C (32°F)."),
    ("How fast does light travel?", "About 299,792,458 metres per second in a vacuum."),
    ("What's the largest ocean on Earth?", "The Pacific."),
    ("Which planet is closest to the sun?", "Mercury."),
    ("Which planet is the largest?", "Jupiter."),
    ("Which planet is known as the red planet?", "Mars, because of iron oxide in its dust."),
    ("How many planets are in the solar system?", "Eight, since Pluto was reclassified as a dwarf planet in 2006."),
    ("How many moons does Mars have?", "Two: Phobos and Deimos."),
    ("What is the sun made of?", "Mostly hydrogen and helium, fusing hydrogen into helium in its core."),
    ("How far is the moon from Earth?", "About 384,400 km on average."),
    ("How many bones are in the adult human body?", "206."),
    ("How many chambers does the human heart have?", "Four."),
    ("What does DNA stand for?", "Deoxyribonucleic acid."),
    ("What is the chemical formula for water?", "H₂O."),
    ("What is the chemical formula for carbon dioxide?", "CO₂."),
    ("What is photosynthesis?", "Plants using sunlight to turn water and CO₂ into sugar and oxygen."),
    ("What is the strongest muscle in the human body?", "It depends how you measure it — the masseter (jaw) wins for bite force."),
    ("How tall is Mount Everest?", "8,849 metres above sea level."),
    ("What galaxy are we in?", "The Milky Way."),
    ("What is the hardest natural substance?", "Diamond."),
    ("What gas do plants take in?", "Carbon dioxide — and they release oxygen."),
    ("How many chromosomes do humans have?", "46, in 23 pairs."),
    ("What is absolute zero?", "0 kelvin, about −273.15°C, where particles have minimum energy."),
    ("What's the speed of sound in air?", "About 343 metres per second at room temperature."),
    ("What's the largest organ in the human body?", "The skin."),
    ("How long does it take light from the sun to reach Earth?", "About 8 minutes 20 seconds."),
    ("What causes tides?", "Mostly the moon's gravity, plus the sun's."),
    ("What is the boiling point of water in Fahrenheit?", "212°F at sea level."),
    ("Are there more stars than grains of sand on Earth?", "Probably — the observable universe has far more stars than Earth has sand grains."),
]
UNITS = [
    ("how many metres in a kilometre", "1,000", "1 km = 1000 m"),
    ("how many metres in a mile", "about 1,609", "1 mile ≈ 1.609 km"),
    ("how many grams in a kilogram", "1,000", "1 kg = 1000 g"),
    ("how many millilitres in a litre", "1,000", "1 L = 1000 ml"),
    ("how many minutes in an hour", "60", "1 h = 60 min"),
    ("how many seconds in a day", "86,400", "24 × 60 × 60 = 86,400"),
    ("how many days in a leap year", "366", "365 plus one extra day in February"),
    ("how many megabytes in a gigabyte", "1,024", "1 GB = 1024 MB in binary units"),
    ("how many bits in a byte", "8", "1 byte = 8 bits"),
    ("how many grams in a metric tonne", "1,000,000", "1 tonne = 1000 kg = 1,000,000 g"),
    ("how many centimetres in a metre", "100", "1 m = 100 cm"),
    ("how many degrees in a right angle", "90", "a quarter turn"),
    ("how many degrees in a triangle's angles", "180", "they always add to 180°"),
    ("how many strings does a standard guitar have", "six", "six strings, E A D G B E"),
]
DEFS = [
    ("algorithm", "a finite set of steps for solving a problem"),
    ("variable", "a name that holds a value in a program"),
    ("function", "a reusable block of code that takes inputs and returns an output"),
    ("compiler", "a program that translates source code into machine code before it runs"),
    ("interpreter", "a program that executes source code directly, line by line"),
    ("bug", "a mistake in code that makes it behave differently from what you intended"),
    ("cache", "a small fast store of data kept to avoid repeating slow work"),
    ("latency", "the delay between asking for something and getting the answer"),
    ("throughput", "how much work gets done per unit of time"),
    ("API", "a defined way for one piece of software to talk to another"),
    ("recursion", "a function calling itself on a smaller version of the problem"),
    ("hash", "a fixed-size fingerprint computed from data"),
    ("encryption", "scrambling data with a key so only the key holder can read it"),
    ("pointer", "a value that stores a memory address"),
    ("token", "a chunk of text — a word, part of a word, or a symbol — that a language model reads"),
    ("embedding", "a vector of numbers representing the meaning of something"),
    ("gradient", "the direction and size of steepest change in a function"),
    ("overfitting", "memorising the training data instead of learning the pattern"),
    ("regex", "a pattern language for matching text"),
    ("git", "a version control system that records snapshots of your code"),
    ("thread", "a single sequence of instructions a program can run"),
    ("database", "an organised store of data you can query"),
    ("pixel", "one coloured dot in an image"),
    ("bandwidth", "how much data can move per second"),
    ("model", "a simplified mathematical description of a system, often trained from data"),
    ("context window", "how much text a model can hold in mind at once"),
    ("quantisation", "storing numbers with fewer bits so a model takes less memory"),
    ("neuron", "a small unit that multiplies inputs by weights and passes on a sum"),
    ("entropy", "a measure of uncertainty or disorder"),
    ("heuristic", "a rule of thumb that usually works but isn't guaranteed"),
]
LANG_FACTS = [
    ("python", "Guido van Rossum", "1991"), ("javascript", "Brendan Eich", "1995"),
    ("java", "James Gosling", "1995"), ("c", "Dennis Ritchie", "1972"),
    ("linux", "Linus Torvalds", "1991"), ("git", "Linus Torvalds", "2005"),
    ("rust", "Graydon Hoare", "2010"), ("php", "Rasmus Lerdorf", "1994"),
]

def build_knowledge():
    for country, cap in CAPITALS:
        q = pick([f"what's the capital of {country.lower()}",
                  f"capital of {country.lower()}?",
                  f"what is the capital city of {country.lower()}",
                  f"do you know the capital of {country.lower()}"])
        a = pick([f"{cap}.", f"{cap}.", f"The capital of {country} is {cap}."])
        if maybe(0.2):
            C("knowledge", q, a, pick(["thanks", "nice one", "cool"]),
              pick(["Anytime.", "Sure.", "No problem — ask me another."]))
        else:
            C("knowledge", q, a)
    for sym, name in ELEMENTS:
        if maybe(0.5):
            q = pick([f"what's the chemical symbol for {name}",
                      f"symbol for {name}?", f"chemical symbol of {name}"])
            a = pick([f"{sym}.", f"{name.capitalize()} is {sym}."])
        else:
            q = pick([f"what element is {sym}", f"what does {sym} stand for in chemistry",
                      f"{sym} is which element"])
            a = pick([f"{name.capitalize()}.", f"{sym} is {name}."])
        C("knowledge", q, a)
    for title, author in BOOKS:
        q = pick([f"who wrote {title}", f"who is the author of {title}",
                  f"{title} was written by who"])
        a = pick([f"{author}.", f"{title} is by {author}."])
        C("knowledge", q, a)
    for q, a in SCIENCE:
        C("knowledge", q.lower() if maybe(0.5) else q,
          a if maybe(0.6) else a.rstrip(".") + ".")
        if maybe(0.25):
            C("knowledge", q, a,
              pick(["how sure are you", "source", "is that exact"]),
              pick(["That's the standard figure — good enough for most purposes, but real"
                    " measurements have error bars.",
                    "I'm confident on the basic fact. If you need precision, check a current"
                    " reference, since exact numbers get refined."]))
    for q, short, why in UNITS:
        C("knowledge", q, f"{short}. ({why})" if maybe(0.5) else f"{short}.")
    for word, meaning in DEFS:
        q = pick([f"what does {word} mean", f"define {word}", f"what is {word}",
                  f"explain {word} to me"])
        a = pick([f"{meaning.capitalize()}.",
                  f"{word.capitalize()} is {meaning}.",
                  f"It means {meaning}."])
        C("language", q, a)
        if maybe(0.18):
            C("language", q, a, pick(["give me an example", "example?"]),
              pick([f"Sure — {EXAMPLE_OF.get(word, 'it shows up everywhere once you know the word')}",
                    f"Simple case: {EXAMPLE_OF.get(word, 'you use it every time you run a program')}"]))
    for lang, author, year in LANG_FACTS:
        C("knowledge", pick([f"who made {lang}", f"who created {lang}", f"who invented {lang}"]),
          f"{author}, in {year}.")

EXAMPLE_OF = {
    "algorithm": "a recipe is an algorithm: ordered steps that turn ingredients into a cake",
    "variable": "`score = 0` stores zero under the name score",
    "function": "`def add(a, b): return a + b` can be called as many times as you like",
    "bug": "an off-by-one loop that runs one time too many",
    "recursion": "a folder scan that calls itself on every subfolder",
    "hash": "two files with the same SHA-256 are almost certainly identical",
    "token": "the word 'unbelievable' might become 'un' + 'believ' + 'able'",
    "embedding": "king minus man plus woman lands near queen in embedding space",
    "cache": "your browser keeping an image on disk instead of re-downloading it",
    "latency": "the pause between pressing a key and seeing the letter appear",
    "overfitting": "a model that memorises the test answers but fails on new questions",
    "regex": "`^\\d{3}-\\d{4}$` matches a phone number like 555-1234",
}

# ----------------------------------------------------------------------------- 3. staying on topic
# The 3M model's worst failure was answering a question it invented. These drills teach the model to
# hold the thread: the user replies tersely ("nothing", "idk", "same"), and the reply must stay on the
# topic the assistant itself just raised, or simply be honest and neutral.
TOPIC_DRILLS = [
    dict(topic="work",
         openers=["what are you working on today", "how's work going", "what's the work situation"],
         asks=["Hey! What's on your plate today?", "Hey. How's the work side of life?",
               "Hi! What are you working on today?"],
         terses=["nothing", "not much", "idk", "same as always", "meh", "nothing really"],
         stays=["Fair enough. Nothing is a valid answer — we can just chat if you want.",
                "Okay. Then let's not talk about work. Anything else on your mind?",
                "Nothing it is. Want a distraction, or are we just hanging out?",
                "Respect. Sometimes the best work update is 'nothing happened yet'."]),
    dict(topic="weekend",
         openers=["what are you up to this weekend", "any weekend plans", "what's the weekend looking like"],
         asks=["Hey! Any plans for the weekend?", "What's the weekend look like for you?"],
         terses=["nothing", "not sure yet", "idk", "dunno", "nope"],
         stays=["A free weekend is underrated. Want ideas, or are you protecting the emptiness?",
                "No plans yet is fine. Do you want something low-key or actually fun?",
                "Then it's wide open. What's your default when nothing is scheduled?",
                "Fair. Some weekends are for doing absolutely nothing, and that's allowed."]),
    dict(topic="music",
         openers=["what are you listening to", "what music are you into", "listening to anything good"],
         asks=["What are you listening to lately?", "Anything good in your headphones?"],
         terses=["nothing", "idk", "same stuff", "not really", "silence"],
         stays=["Silence is a genre too. Want something new to try?",
                "Fair. What's the last thing that actually stuck with you?",
                "Okay — then I'll ask the important question: one song you never skip?",
                "No music at the moment is a valid state. Want me to suggest something?"]),
    dict(topic="food",
         openers=["what are you eating", "what's for dinner", "had anything good to eat"],
         asks=["What are you eating today?", "What's for dinner?"],
         terses=["nothing", "idk", "not hungry", "nothing yet"],
         stays=["Nothing yet is fixable. Want something quick, or something worth cooking?",
                "Fair. What's in the fridge? I can work with that.",
                "Then I'll ask the useful question: how much effort are you willing to spend on food right now?",
                "Okay. What's the laziest meal you actually enjoy?"]),
    dict(topic="code",
         openers=["what are you building", "what are you coding", "working on anything fun"],
         asks=["What are you building right now?", "Any code on your plate today?"],
         terses=["nothing", "idk", "no", "stuck", "procrastinating"],
         stays=["Stuck is a real state. What's the last thing that worked?",
                "Procrastinating is a mood. Want a five-minute task to break the freeze?",
                "Nothing's fine. If you want a small puzzle to warm up, say the word.",
                "Fair. What's the project you keep opening and closing?"]),
    dict(topic="sleep",
         openers=["how did you sleep", "you tired", "sleep ok"],
         asks=["How did you sleep?", "You look tired — did you get any sleep?"],
         terses=["bad", "not really", "idk", "no", "barely"],
         stays=["Rough night? Do you want practical fixes or just sympathy?",
                "Barely counts as zero. If you can, keep today light — tired decisions are usually bad ones.",
                "That's brutal. What kept you up?",
                "Okay. One thing that helps tonight is not chasing the lost sleep — just go to bed at a normal time."]),
    dict(topic="mood",
         openers=["how are you feeling", "you ok", "what's your mood"],
         asks=["How are you actually feeling?", "You doing okay?"],
         terses=["nothing", "idk", "fine", "meh", "same"],
         stays=["\"Fine\" is a suspicious word. Anything under it?",
                "Okay. I'm not going to push. If you want to talk about something else, we can.",
                "Meh is honest. Want distraction or company?",
                "Fair enough. What would make today one notch better?"]),
    dict(topic="projects",
         openers=["what's your side project", "working on anything personal", "building anything"],
         asks=["What are you building on the side?", "Any personal project going?"],
         terses=["nothing", "idk", "gave up", "on pause"],
         stays=["Paused isn't dead. What made you stop?",
                "Gave up on which one? I bet it's still salvageable.",
                "Nothing is allowed. But if you had to pick one idea to revive, what would it be?",
                "On pause is fine. What was the best part of it?"]),
    dict(topic="study",
         openers=["studying for anything", "what are you learning", "any exams coming"],
         asks=["Are you studying for anything right now?", "What are you learning these days?"],
         terses=["nothing", "idk", "not really", "nothing much"],
         stays=["Then I'll ask a better question: what do you wish you understood better?",
                "Fair. If you want to learn something small today, I'll pick a topic and quiz you.",
                "Nothing is fine. Learning droughts happen.",
                "Okay, different angle: what's something you're curious about but never looked up?"]),
]

def build_topic_drills(n_target):
    made = 0
    while made < n_target:
        t = pick(TOPIC_DRILLS)
        opener = pick(["hi", "hey", "hello", "yo", "hey there", "hi again", "sup"])
        convo = [opener, pick(t["asks"]), pick(t["terses"]), pick(t["stays"])]
        if maybe(0.30):
            extra_user = pick(["yeah sure", "ok", "maybe", "why not", "nah", "true"])
            extra_ai = pick([
                "Alright. Give me one word for how the day's going and I'll take it from there.",
                "Then here's an easy one: what's something good that happened this week?",
                "Okay, I'll go first — I like the sound of rain on a window. Your turn.",
                "Cool. What are you avoiding right now?"
            ])
            convo += [extra_user, extra_ai]
        C("conversation", *convo)
        made += 1

# ----------------------------------------------------------------------------- 4. corrections
def build_corrections(n_target):
    made = 0
    while made < n_target:
        a, b = RNG.randint(2, 12), RNG.randint(2, 12)
        r = a * b
        wrong = r + pick([-2, -1, 1, 2, 10 if r < 50 else -10])
        q = pick([f"what's {a} times {b}", f"{a} * {b}", f"what is {a} x {b}"])
        if maybe(0.6):
            C("correction", q, f"{r}.",
              pick([f"no it's {wrong}", f"wrong, it's {wrong}", f"that's {wrong} not {r}",
                    f"i think you mean {wrong}"]),
              pick([f"Let me recheck: {a} × {b} = {r}. I'm confident in that one.",
                    f"I get {r} again. {a} × {b} = {r}.",
                    f"Checked twice: {a} × {b} = {r}, so {wrong} would be off by {abs(wrong - r)}."]))
        else:
            C("correction", q, f"{r}.",
              pick(["no", "nah", "wrong"]),
              pick([f"Then tell me what you're seeing. {a} × {b} = {r} here.",
                    f"Which part looks off? {a} × {b} = {r}.",
                    f"Happy to redo it — but {a} × {b} is {r}."]))
        made += 1
    for country, cap in CAPITALS[:40]:
        wrong = pick([c for c in ["Sydney", "Toronto", "Rio de Janeiro", "Istanbul", "Mumbai",
                                  "Lagos", "Barcelona", "Geneva", "Kyoto", "Dubai"]
                      if c != cap])
        C("correction", f"what's the capital of {country.lower()}", f"{cap}.",
          pick([f"no it's {wrong}", f"isn't it {wrong}", f"pretty sure it's {wrong}"]),
          pick([f"{wrong} is the bigger city, but {cap} is the capital.",
                f"Common mix-up. {cap} is the capital — {wrong} isn't.",
                f"No — {cap}. {wrong} gets confused with it a lot."]))
    # user corrects themselves / clarifies a typo
    for _ in range(max(0, n_target - made - 40)):
        a, b = RNG.randint(2, 9), RNG.randint(2, 9)
        r = a * b
        C("correction", pick([f"what's {a+1} times {b}", f"{a+1} * {b}"]),
          f"{(a + 1) * b}.",
          pick(["sorry i meant {a} times {b}", "typo, {a} x {b}", "oops i meant {a} * {b}"]).format(a=a, b=b),
          pick([f"Then it's {r}.", f"{a} × {b} = {r}.", f"Fixed: {r}."]))
        made += 1

# ----------------------------------------------------------------------------- 5. memory + follow-ups
MEM_ITEMS = [
    ("i have two cats, mika and juno", "how many cats do i have", "Two — Mika and Juno."),
    ("my name is {name}", "what's my name", "{Name}."),
    ("i live in {city}", "where do i live", "{city}."),
    ("i'm {age} years old", "how old am i", "{age} — you told me earlier."),
    ("i'm building a game about lighthouses", "what's my game about", "Lighthouses."),
    ("i hate cilantro", "what food do i hate", "Cilantro."),
    ("i work as a nurse", "what's my job", "You're a nurse."),
    ("my favourite band is radiohead", "what's my favourite band", "Radiohead."),
    ("i'm learning guitar", "what instrument am i learning", "Guitar."),
    ("i have a dog called pepper", "what's my dog's name", "Pepper."),
    ("i'm allergic to peanuts", "what am i allergic to", "Peanuts."),
    ("i moved to wellington last year", "where did i move to", "Wellington."),
]
CITIES = ["auckland", "wellington", "berlin", "lisbon", "toronto", "seoul", "bristol", "perth"]

def build_memory(n_target):
    made = 0
    while made < n_target:
        item = pick(MEM_ITEMS)
        name = pick(NAMES).capitalize()
        city = pick(CITIES).capitalize()
        age = RNG.randint(18, 60)
        say = item[0].format(name=name, city=city, age=age)
        ask = item[1].format(name=name, city=city, age=age)
        ans = item[2].format(Name=name, city=city, age=age)
        ack = pick(["Got it.", "Noted.", "Okay, noted.", "Good to know."])
        if maybe(0.5):
            C("memory", say, ack, ask, ans)
        else:
            filler_q = pick(["what's up", "how's your day going", "hey", "random question: coffee or tea?"])
            filler_a = pick(["Not much — what about you?", "Pretty quiet. You?",
                             "Good. What's on your mind?", "Tea, obviously. You?"])
            C("memory", say, ack, filler_q, filler_a, ask, ans)
        made += 1
    # summaries of the conversation
    for _ in range(max(1, n_target // 8)):
        n1, c1 = pick(NAMES).capitalize(), pick(CITIES).capitalize()
        job = pick(["designer", "teacher", "nurse", "developer", "chef", "student"])
        budget = pick([100, 250, 400, 800])
        C("memory", f"i'm {n1}, i'm a {job} in {c1} and i'm trying to save {budget} a month",
          pick([f"Good to meet you, {n1}. Saving {budget} a month as a {job} in {c1} is doable "
                f"if we pick one thing to cut rather than ten. What's the biggest monthly cost right now?",
                f"Nice to meet you, {n1}. Let's make the {budget} realistic: what's your biggest "
                f"fixed cost, rent or transport?"]),
          "what do you know about me so far",
          f"You're {n1}, a {job} in {c1}, aiming to save {budget} a month.")
    # user changes their mind mid-conversation
    for _ in range(max(1, n_target // 6)):
        col1, col2 = pick([("green", "blue"), ("red", "orange"), ("black", "navy")])
        C("memory", f"my favourite colour is {col1}", "Noted.",
          f"actually it's {col2}", f"Updated — {col2}. I'll go with that.")

# ----------------------------------------------------------------------------- 6. reasoning
DAYNAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def build_reasoning(n_target):
    made = 0
    while made < n_target:
        kind = RNG.randint(0, 9)
        if kind == 0:  # compare two values
            fr = pick([(3, 4), (2, 3), (5, 8), (7, 10), (1, 3), (5, 6), (2, 5), (9, 10), (3, 8), (4, 5)])
            dec = pick([0.7, 0.75, 0.6, 0.66, 0.5, 0.8, 0.4, 0.9, 0.35, 0.62])
            val = fr[0] / fr[1]
            bigger = f"{fr[0]}/{fr[1]}" if val > dec else str(dec)
            smaller = str(dec) if val > dec else f"{fr[0]}/{fr[1]}"
            C("reasoning", pick([f"which is bigger, {fr[0]}/{fr[1]} or {dec}",
                                 f"is {fr[0]}/{fr[1]} bigger than {dec}",
                                 f"{fr[0]}/{fr[1]} vs {dec} — which wins"]),
              f"{bigger}. {fr[0]}/{fr[1]} is {val:.3f}, and {dec} is {'smaller' if val > dec else 'bigger'}"
              f" than that.")
        elif kind == 1:  # sequence
            typ = RNG.randint(0, 2)
            if typ == 0:
                start, step = RNG.randint(1, 9), RNG.randint(2, 9)
                seq = [start + step * i for i in range(4)]
                nxt = seq[-1] + step
                why = f"each term adds {step}"
            elif typ == 1:
                start = RNG.randint(1, 4)
                seq = [start * 2 ** i for i in range(4)]
                nxt = seq[-1] * 2
                why = "each term doubles"
            else:
                base = RNG.randint(1, 5)
                seq = [base, (base + 1) ** 2, (base + 2) ** 2, (base + 3) ** 2]
                nxt = (base + 4) ** 2
                why = f"these are {base}², {base+1}², {base+2}², {base+3}²"
            C("reasoning", pick([f"what comes next: {', '.join(map(str, seq))}",
                                 f"continue the sequence {', '.join(map(str, seq))}",
                                 f"{', '.join(map(str, seq))} ... what's next"]),
              f"{nxt}. {why}.")
        elif kind == 2:  # letters in a word
            word = pick(["strawberry", "banana", "mississippi", "parallel", "assessment",
                         "committee", "necessary", "bookkeeper", "rhythm", "success"])
            letter = pick(sorted(set(word)))
            count = word.count(letter)
            C("reasoning", pick([f"how many {letter}'s are in {word}",
                                 f"count the letter {letter} in {word}"]),
              f"{count}. {word} spelled out: {'-'.join(word)}.")
        elif kind == 3:  # day of week
            start = pick(DAYNAMES)
            add = RNG.randint(2, 40)
            idx = (DAYNAMES.index(start) + add) % 7
            C("reasoning", pick([f"if today is {start}, what day is it in {add} days",
                                 f"{start} plus {add} days is which day"]),
              f"{DAYNAMES[idx]}. {add} days is {add % 7} past a full week from {start}.")
        elif kind == 4:  # clock
            h, m = RNG.randint(1, 11), pick([0, 15, 30, 45])
            add = pick([45, 90, 150, 200])
            total = h * 60 + m + add
            nh, nm = (total // 60) % 12 or 12, total % 60
            C("reasoning", pick([f"it's {h}:{m:02d}. what time will it be in {add} minutes",
                                 f"start at {h}:{m:02d} and add {add} minutes — what time is it"]),
              f"{nh}:{nm:02d}. {add} minutes is {add // 60}h {add % 60}m later.")
        elif kind == 5:  # age difference
            me, sis = RNG.randint(8, 16), RNG.randint(2, 7)
            now = me + RNG.randint(5, 25)
            C("reasoning", pick([f"i was {me} when my sister was {sis}. now i'm {now}. how old is she",
                                 f"when i was {me} my sister was {sis}; i'm {now} now, so how old is she"]),
              f"{now - (me - sis)}. You're {me - sis} years older, so she's {now} - {me - sis} = "
              f"{now - (me - sis)}.")
        elif kind == 6:  # transitive logic
            a, b, c = pick([("cats", "mammals", "animals"), ("roses", "flowers", "plants"),
                            ("sparrows", "birds", "animals"), ("oaks", "trees", "plants"),
                            ("sedans", "cars", "vehicles")])
            C("reasoning", f"all {a} are {b} and all {b} are {c}. are all {a} {c}",
              f"Yes. If every {a[:-1]} is a {b[:-1]}, and every {b[:-1]} is a {c[:-1]}, then every "
              f"{a[:-1]} is a {c[:-1]}.")
        elif kind == 7:  # unit conversion
            km = RNG.randint(2, 40)
            C("reasoning", pick([f"convert {km} kilometres to metres",
                                 f"how many metres is {km} km"]),
              f"{num(km * 1000)} metres.")
        elif kind == 8:  # ordering
            xs = RNG.sample(range(2, 99), 5)
            order = "ascending" if maybe(0.5) else "descending"
            srt = sorted(xs) if order == "ascending" else sorted(xs, reverse=True)
            C("reasoning", f"sort these {order}: {', '.join(map(str, xs))}",
              f"{', '.join(map(str, srt))}.")
        else:  # odd one out
            group, outsider = pick([(["apple", "pear", "banana"], "carrot"),
                                    (["dog", "cat", "horse"], "table"),
                                    (["red", "blue", "green"], "seven"),
                                    (["hammer", "saw", "drill"], "sock"),
                                    (["2", "4", "8"], "9")])
            items = group + [outsider]
            RNG.shuffle(items)
            C("reasoning", f"which one doesn't belong: {', '.join(items)}",
              f"{outsider} — the others are {pick(['all the same kind of thing', 'in one category'])}. "
              f"{outsider} isn't.")
        made += 1

# ----------------------------------------------------------------------------- 7. style control
STYLE_BASE = [
    ("photosynthesis", "the process where plants use sunlight to turn water and carbon dioxide into sugar and oxygen",
     "plants eating light", "Plants use sunlight to make food from water and air, and they breathe out oxygen."),
    ("a function", "a reusable block of code that takes inputs and returns an output",
     "a machine in a factory that takes something in and spits something out", "A function is a named chunk of code you can run whenever you want, with inputs and one output."),
    ("inflation", "the rate at which prices rise over time, which reduces what money can buy",
     "your pocket money buying less each year", "Prices creep up, so the same amount of money buys less than it used to."),
    ("gravity", "the attraction between things that have mass, which curves spacetime",
     "the floor pulling on everything", "Gravity is why things fall down and why planets stay in orbit."),
    ("a tokenizer", "the component that splits text into the small chunks a model reads",
     "a machine that cuts sentences into LEGO bricks", "A tokenizer chops text into pieces — words, word parts or symbols — before the model sees it."),
    ("latency", "the delay before a response starts, measured in time",
     "the wait before the rollercoaster moves", "Latency is the pause between asking and getting an answer."),
    ("recursion", "a function that solves a problem by calling itself on a smaller version of it",
     "standing between two mirrors and seeing yourself forever", "Recursion is when a thing contains a smaller copy of itself and stops at a base case."),
    ("encryption", "scrambling data with a key so that only someone holding the key can read it",
     "a lockbox only your friend has the key to", "Encryption turns readable data into noise that only the right key can turn back."),
    ("overfitting", "when a model memorises the training data instead of learning the pattern",
     "memorising last year's exam answers and failing the new paper", "Overfitting is a model that's great at what it saw and bad at everything else."),
    ("entropy", "a measure of how many ways a system's parts can be arranged",
     "the reason your room gets messy without you trying", "Entropy is why mess appears on its own and tidying takes effort."),
    ("bandwidth", "how much data can move through a connection per second",
     "how wide the pipe is, not how long the water takes", "Bandwidth is how much data fits through at once."),
    ("a database", "an organised store of data you can query quickly",
     "a filing cabinet that can find any page instantly", "A database stores data in a way that makes asking questions about it fast."),
    ("a closure", "a function that remembers the variables from where it was created",
     "a backpack a function carries around", "A closure is a function plus the variables it kept from the place it was born."),
    ("a hash", "a fixed-length fingerprint computed from data",
     "a fingerprint for a file", "A hash turns data into a short fingerprint that changes completely if the data changes."),
    ("a cache", "a small fast store of recently used data",
     "the snack drawer next to your desk", "A cache keeps recent answers close so you don't recompute them."),
    ("inflation (in simple terms)", "prices creeping up so money buys less over time",
     "your pocket money buying less each year", "Prices go up, so the same money buys less than before."),
    ("an operating system", "the software that manages hardware and runs your programs",
     "the manager of a building everyone works in", "An operating system is the layer that lets programs use the hardware without fighting each other."),
]
def build_style(n_target):
    made = 0
    while made < n_target:
        word, defn, analogy, simple = pick(STYLE_BASE)
        mode = RNG.randint(0, 4)
        if mode == 0:
            C("style", f"explain {word} like i'm 5", f"{analogy.capitalize()}. {simple}")
        elif mode == 1:
            C("style", f"explain {word} in one sentence", f"{defn.capitalize()}.")
        elif mode == 2:
            C("style", f"explain {word}", f"{defn.capitalize()}.", pick(["more detail", "go deeper", "and?"]),
              f"{simple} More precisely: {defn}.")
        elif mode == 3:
            C("style", f"give me the tl;dr on {word}", f"{simple}")
        else:
            C("style", f"what is {word}", f"{defn.capitalize()}.",
              pick(["shorter", "too long", "one line"]),
              f"{simple}")
        made += 1
    # compressed / expanded answers
    for _ in range(max(1, n_target // 3)):
        C("style", pick(["can you be more concise from now on", "keep it short from now on",
                         "be brief please"]),
          pick(["Done — short answers unless you ask for depth.",
                "Got it. I'll keep it tight.",
                "Sure. Short by default, detail on request."]),
          "what's 12 + 30",
          "42.")
        C("style", pick(["give me more detail next time", "i like long answers",
                         "explain things fully"]),
          pick(["Understood — I'll go deeper unless you tell me to cut it.",
                "Fine by me. I'll explain the reasoning, not just the answer."]),
          "what's 12 + 30",
          "42. That's 12 + 30: ten plus thirty is forty, then the two left over makes 42.")

# ----------------------------------------------------------------------------- 8. coding
CODE_QA = [
    ("what does this do\n\nfor i in range(3):\n    print(i)",
     "It prints 0, 1, 2 — one number per line. `range(3)` counts from 0 up to but not including 3."),
    ("why does this fail\n\nprint('hello' + 5)",
     "You can't add a string and an int. Use `print('hello ' + str(5))` or an f-string: `print(f'hello {5}')`."),
    ("write a function that reverses a string in python",
     "```python\ndef reverse(s):\n    return s[::-1]\n```\n`[::-1]` slices with a step of -1, so it walks the string backwards."),
    ("how do i read a file in python",
     "```python\nwith open('data.txt') as f:\n    text = f.read()\n```\n`with` closes the file for you, even if something throws."),
    ("what's a list comprehension",
     "A compact way to build a list. `[x * 2 for x in range(5)]` gives `[0, 2, 4, 6, 8]`."),
    ("how do i swap two variables in python",
     "`a, b = b, a` — Python unpacks the right side before assigning, so you don't need a temp variable."),
    ("what does len() do",
     "Returns the number of items: length of a string, list, dict or tuple. `len('abc')` is 3."),
    ("difference between a list and a tuple",
     "Lists are mutable (you can change them), tuples are immutable. Tuples are cheaper and can be dict keys."),
    ("how do i check if a key is in a dict",
     "`if 'key' in d:` — that's the idiomatic way, and it's fast."),
    ("why is my loop infinite",
     "Something inside the loop never changes the condition. If you're using `while i < 10`, check that `i += 1` actually runs."),
    ("how do i convert a string to an int",
     "`int('42')` gives 42. It raises ValueError if the string isn't a number, so wrap it in try/except if the input is untrusted."),
    ("what's the difference between == and is",
     "`==` compares values, `is` compares identity (same object in memory). Use `is` only for None, True and False."),
    ("how do i sort a list of dicts by a key",
     "```python\nitems.sort(key=lambda d: d['score'])\n```\nAdd `reverse=True` for descending."),
    ("what's a closure",
     "A function that remembers variables from the scope where it was created, even after that scope ends."),
    ("how do i format a number to two decimals",
     "`f'{x:.2f}'` in an f-string, or `round(x, 2)` if you want a number back."),
    ("what does try/except do",
     "It runs code in `try`; if an exception happens, `except` catches it instead of crashing the program."),
    ("how do i get the current time in python",
     "```python\nfrom datetime import datetime\nnow = datetime.now()\n```"),
    ("what's the difference between / and // in python",
     "`/` is true division and returns a float. `//` is floor division and rounds down to an integer."),
    ("write fizzbuzz",
     "```python\nfor i in range(1, 101):\n    if i % 15 == 0:\n        print('FizzBuzz')\n    elif i % 3 == 0:\n        print('Fizz')\n    elif i % 5 == 0:\n        print('Buzz')\n    else:\n        print(i)\n```"),
    ("how do i remove duplicates from a list",
     "`list(set(items))` removes duplicates but loses order. To keep order: `list(dict.fromkeys(items))`."),
    ("what's a generator in python",
     "A function using `yield` that produces values lazily, one at a time, instead of building a whole list."),
    ("how do i join a list into a string",
     "`', '.join(items)` — the separator goes before `.join`, and every item must be a string."),
    ("how do i check a type in python",
     "`isinstance(x, int)` — better than `type(x) == int` because it respects subclasses."),
    ("what's a lambda",
     "An anonymous one-expression function. `lambda x: x + 1` is the same as a def that returns x + 1."),
    ("how do i parse json in python",
     "```python\nimport json\ndata = json.loads(text)   # string -> dict\n```\nUse `json.load(f)` to read from a file."),
    ("what's recursion",
     "A function that calls itself on a smaller version of the problem, with a base case that stops it."),
    ("how do i make a http request in python",
     "```python\nimport urllib.request\nwith urllib.request.urlopen(url) as r:\n    body = r.read()\n```\nOr `pip install requests` and use `requests.get(url)`."),
    ("explain async await",
     "`async` marks a function that can pause. `await` waits for an async call without blocking everything else, so other work can run meanwhile."),
    ("what's the difference between let and const in javascript",
     "Both are block-scoped. `const` can't be reassigned; `let` can. Objects inside a `const` can still be mutated."),
    ("how do i centre a div",
     "```css\n.parent { display: flex; justify-content: center; align-items: center; }\n```\nThat centres the child both ways."),
]
def build_coding(n_target):
    made = 0
    while made < n_target:
        q, a = pick(CODE_QA)
        prefix = pick(["", "", "quick q: ", "hey, ", "one more: "])
        C("coding", prefix + q, a)
        if maybe(0.2):
            C("coding", q, a, pick(["thanks", "nice", "that worked"]),
              pick(["Anytime.", "Good. Shout if it breaks.", "Nice."]))
        made += 1

# ----------------------------------------------------------------------------- 9. advice
ADVICE = [
    ("i can't sleep", ["try lying still with your eyes closed and let your breathing slow down — "
                       "rest still counts even if sleep doesn't come. If your brain is looping, "
                       "write the loop down on paper so it stops rehearsing.",
                       "get out of bed for ten minutes if you're properly awake. Do something boring "
                       "in dim light, then go back. Beds should mean sleep, not struggle."]),
    ("how do i stop procrastinating", ["shrink the task until it's almost insulting: two minutes, "
                                       "one file open, no goal beyond starting. Starting is the hard "
                                       "part; momentum does the rest.",
                                       "pick the smallest visible piece and do it badly on purpose. "
                                       "You can fix bad work — you can't fix a blank page."]),
    ("how do i focus better", ["one window, one task, one timer. Twenty-five minutes of single-tasking "
                               "beats two hours of half-attention, and phone-in-another-room is worth "
                               "more than any app.",
                               "decide what you're doing before you sit down. Vague intentions turn "
                               "into tab-hopping."]),
    ("i'm stressed about money", ["write down what actually comes in and out this month — most money "
                                  "stress is bigger than the numbers once they're visible. Then pick "
                                  "one recurring cost to cut, not five.",
                                  "separate 'can't pay' from 'don't like paying'. Only the first needs "
                                  "action today."]),
    ("how do i get better at running", ["mostly slow miles, one hard session a week, and don't add "
                                        "more than about 10% distance weekly. Consistency beats "
                                        "heroic single runs.",
                                        "run slower than feels right for most sessions. You should be "
                                        "able to talk. Speed comes from the occasional hard day."]),
    ("how do i prepare for a job interview", ["write five stories about things you actually did, each "
                                              "with a result attached, and practise saying them out "
                                              "loud. Most interview questions are one of those five in "
                                              "disguise.",
                                              "research the team, not just the company, and prepare two "
                                              "questions that prove you thought about the work."]),
    ("i'm burnt out", ["cut the load before you try to optimise it. Burnout isn't a discipline problem, "
                       "it's an empty tank, and rest is the repair, not the reward.",
                       "take the smallest real break you can defend — a day with no obligations — and "
                       "tell someone at work what's happening. Hiding it makes it worse."]),
    ("how do i build a habit", ["make it tiny, attach it to something you already do, and never miss "
                                "twice. Two missed days is how habits die.",
                                "track the streak somewhere visible. The streak does the motivation "
                                "for you after about two weeks."]),
    ("how do i stop arguing with my partner", ["drop the goal of winning. Try 'what do we both want "
                                               "here' instead of 'who's right', and take a twenty minute "
                                               "break before either of you says something expensive.",
                                               "argue about the specific thing, not the pattern. 'You "
                                               "always' turns a small problem into a war."]),
    ("how do i learn to code", ["pick one small thing to build, then learn only what it needs. "
                                "Tutorials feel productive but building is where it sticks.",
                                "type every example yourself instead of reading it, and break it on "
                                "purpose to see the error messages. Errors are the curriculum."]),
]
ADVICE_Q = ["what should i do", "any advice", "help", "what would you do", "thoughts"]

def build_advice(n_target):
    made = 0
    while made < n_target:
        topic, answers = pick(ADVICE)
        q = topic if maybe(0.7) else f"{topic}, {pick(ADVICE_Q)}"
        a = pick(answers)
        if maybe(0.35):
            follow = pick(["thanks", "ok i'll try", "that helps", "makes sense"])
            back = pick(["Good. One small step today beats a plan you never start.",
                         "That's the whole trick — keep it small enough to actually do.",
                         "You've got this. Tell me how it goes.",
                         "Start with the smallest version and adjust after."])
            C("advice", q, a, follow, back)
        else:
            C("advice", q, a)
        made += 1

# ----------------------------------------------------------------------------- 10. ideas & lists
IDEAS = {
    "coffee shop": ["The Slow Drip", "Bitter Hours", "Two Sugars", "The Daily Grind", "Static & Steam"],
    "band": ["Paper Lanterns", "Static Youth", "The Quiet Hours", "Neon Fields", "Low Tide Radio"],
    "cat": ["Mochi", "Pixel", "Nimbus", "Sable", "Biscuit", "Juniper", "Comet"],
    "dog": ["Pepper", "Scout", "Waffles", "Marlow", "Juno", "Tofu"],
    "coding project": ["a CLI that renames files by date", "a tiny static site generator",
                       "a local-first notes app", "a bot that summarises your git log",
                       "a terminal pomodoro timer"],
    "weekend": ["a long walk with a podcast and no destination", "cook something you've never made",
                "clear one drawer entirely", "go to the library and pick a random shelf",
                "message someone you've been meaning to"],
    "birthday gift": ["a book they'd never buy themselves", "a really good pen",
                      "a framed photo you took", "a voucher for something they keep talking about",
                      "a plant with an absurd name"],
    "dinner": ["pasta with garlic, chilli and lemon", "rice, egg and whatever greens are dying",
               "tinned tomatoes, beans and toast", "noodles with peanut butter and soy",
               "roast vegetables with yoghurt and herbs"],
}
def build_ideas(n_target):
    made = 0
    while made < n_target:
        thing, options = pick(list(IDEAS.items()))
        k = RNG.randint(3, min(5, len(options)))
        chosen = options[:k]
        q = pick([f"give me {k} ideas for a {thing}", f"{k} names for a {thing}",
                  f"i need {k} ideas for {thing}s", f"suggest {k} things for a {thing}"])
        if thing in ("cat", "dog", "band", "coffee shop"):
            a = "Some options:\n" + "\n".join(f"- {c}" for c in chosen)
        else:
            a = "Here are a few:\n" + "\n".join(f"- {c}" for c in chosen)
        if maybe(0.3):
            C("advice", q, a, pick(["more", "another", "none of those"]),
              pick(["Then let's go sideways: take the one you hated least and change one word.",
                    "Fair. Tell me the vibe you want and I'll aim closer.",
                    "Different angle: what do you want someone to feel when they see it?"]))
        else:
            C("advice", q, a)
        made += 1

# ----------------------------------------------------------------------------- 11. boundaries
SAFETY = [
    ("how do i pick a lock", "I'll skip that one — lock-picking instructions aren't something I'll give out. If you're locked out of your own place, a locksmith or your building manager is the fast route."),
    ("how do i hack someone's instagram", "No — that's breaking into someone's account and I won't help with it. If it's your own account, use the official recovery flow."),
    ("write malware for me", "I won't write malware. If you're learning security, I'm happy to explain how attacks work defensively, or help you set up a practice lab."),
    ("how do i make a bomb", "Not something I'll help with. If you're curious about the chemistry, I can point you at textbooks that stay theoretical."),
    ("how do i cheat on my exam", "I'll pass on that. I can help you actually learn the material in a fraction of the time though — what's the subject?"),
    ("help me write a fake review for my competitor", "That's deception aimed at hurting someone's business, so no. I can help you get real reviews or improve what's actually weak."),
    ("how do i get revenge on my ex", "I get the impulse, but I'm not going to help with revenge — it usually costs more than it pays. Want to talk about what actually happened?"),
    ("can you write something racist", "No. I'll write plenty of things, but not that."),
    ("how do i lie to my boss about being sick", "I'd rather not script that. If you need a day off, 'I need to take a day' is usually enough — want help saying it plainly?"),
    ("tell me how to cook meth", "Definitely not. I'll help with normal cooking any time."),
    ("how do i hurt someone", "No. Not that. If you're angry at someone specific and want to handle it without wrecking things, I'll help with that."),
    ("i want to kill myself", "I'm really glad you said it out loud. Please talk to someone who can be with you right now — in New Zealand you can call or text 1737 any time, and 111 if you're in immediate danger. If you're elsewhere, search for your local crisis line. I'm still here if you want to keep talking."),
    ("how do i stalk someone without them knowing", "No. That's surveillance of someone who hasn't consented to it. If you're struggling after a breakup, that's worth talking about instead."),
    ("give me someone's home address", "I don't have that and wouldn't share it. I only know what's in this conversation."),
    ("how do i plagiarise an essay", "I won't write something for you to hand in as yours. I will help you outline it, check your reasoning, and tighten your own writing."),
]
def build_safety():
    for q, a in SAFETY:
        C("safety", q, a)
        if maybe(0.4):
            C("safety", q, a, pick(["ok", "fine", "whatever", "why not"]),
              pick(["Because it would hurt someone, and that's enough of a reason.",
                    "It's a hard line, not a mood. Anything else I can help with?",
                    "Because I'd rather be useful than agreeable.", ]))

# ----------------------------------------------------------------------------- 12. language tasks
SPELL = [("necessary", "neccessary"), ("definitely", "definately"), ("separate", "seperate"),
         ("receive", "recieve"), ("because", "becuase"), ("accommodation", "accomodation"),
         ("tomorrow", "tommorow"), ("rhythm", "rythm"), ("beautiful", "beutiful"),
         ("weird", "wierd"), ("occurred", "occured"), ("embarrass", "embarass"),
         ("privilege", "priviledge"), ("gauge", "guage"), ("conscious", "concious"),
         ("maintenance", "maintainance"), ("millennium", "millenium"), ("broccoli", "brocolli"),
         ("recommend", "reccomend"), ("argument", "arguement")]
def build_language(handled_defs):
    for correct, wrong in SPELL:
        C("language", pick([f"is it {correct} or {wrong}", f"how do you spell {correct}",
                            f"{correct} or {wrong}?"]),
          f"{correct}.")
    GRAMMAR = [
        ("me and him went to the store", "He and I went to the store."),
        ("their going to the park", "They're going to the park."),
        ("i could of done it", "I could have done it."),
        ("she don't like it", "She doesn't like it."),
        ("the dogs bone", "The dog's bone."),
        ("there are less people here today", "There are fewer people here today."),
        ("i seen it yesterday", "I saw it yesterday."),
        ("between you and I", "Between you and me."),
        ("he did good on the test", "He did well on the test."),
        ("alot of people came", "A lot of people came."),
    ]
    for bad, good in GRAMMAR:
        C("language", pick([f"fix this: {bad}", f"is this correct: {bad}", f"correct my grammar: {bad}"]),
          f"{good}" + (f" ({pick(['the pronoun order is the issue', 'the apostrophe was missing', 'wrong word there'])})"
                       if maybe(0.3) else ""))
    TRANS = [
        ("hello, how are you", "Hola, ¿cómo estás?", "spanish"),
        ("thank you very much", "Muchas gracias.", "spanish"),
        ("where is the station", "¿Dónde está la estación?", "spanish"),
        ("i would like a coffee", "Quisiera un café.", "spanish"),
        ("good morning", "Bonjour.", "french"),
        ("thank you", "Merci.", "french"),
        ("where is the bathroom", "Où sont les toilettes ?", "french"),
        ("i don't understand", "Je ne comprends pas.", "french"),
        ("good evening", "Guten Abend.", "german"),
        ("thank you very much", "Vielen Dank.", "german"),
        ("i am hungry", "Ich habe Hunger.", "german"),
        ("see you tomorrow", "Bis morgen.", "german"),
    ]
    for en, tr, lang in TRANS:
        C("language", pick([f"how do you say '{en}' in {lang}", f"translate '{en}' to {lang}",
                            f"'{en}' in {lang}?"]), f"{tr}")

# ----------------------------------------------------------------------------- 13. hand-written data
def load_handwritten():
    """Parse scripts/data/*.txt:  '#category' starts a conversation, then 'U:'/'A:' lines."""
    d = os.path.join(HERE, "data")
    if not os.path.isdir(d):
        return 0
    total = 0
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".txt"):
            continue
        cat, msgs = "conversation", []
        for raw in open(os.path.join(d, fn), encoding="utf-8"):
            line = raw.rstrip("\n").strip()
            if not line:
                continue
            if line.startswith("#"):
                if msgs and len(msgs) % 2 == 0:
                    CONVS.append({"category": cat, "messages": [dict(m) for m in msgs]})
                    total += 1
                cat, msgs = line[1:].strip() or "conversation", []
                continue
            m = re.match(r"^(U|A):\s*(.*)$", line)
            if not m:
                print(f"  ! unparsed line in {fn}: {line[:60]}", file=sys.stderr)
                continue
            msgs.append({"role": "user" if m.group(1) == "U" else "assistant",
                         "content": m.group(2).strip()})
        if msgs and len(msgs) % 2 == 0:
            CONVS.append({"category": cat, "messages": [dict(m) for m in msgs]})
            total += 1
    return total

# ----------------------------------------------------------------------------- 16. explanations
EXPLAIN = [
    ("why is the sky blue", "Sunlight contains every colour. Air molecules scatter the short wavelengths — "
     "blue and violet — far more than red, so blue light reaches your eye from all over the sky instead of "
     "just from the sun."),
    ("why is the ocean salty", "Rain wears minerals out of rock and carries them to the sea. Water leaves "
     "by evaporation, but the salt stays behind, so it builds up over millions of years."),
    ("how does a fridge work", "It moves heat from inside to outside rather than creating cold. A refrigerant "
     "absorbs heat as it evaporates inside, then dumps that heat outside as it's compressed back into a liquid."),
    ("why do we dream", "No one knows for sure. The leading ideas are that dreams help consolidate memories, "
     "rehearse threats safely, and stitch together fragments of the day while your brain runs offline."),
    ("how does wifi work", "A router converts data into radio waves and broadcasts them on specific "
     "frequencies. Your device's antenna picks up the waves and turns them back into data — all in "
     "fractions of a second."),
    ("why is the sky red at sunset", "At sunset light travels through much more atmosphere, so most of the "
     "blue is scattered away and the longer red and orange wavelengths get through."),
    ("how do batteries work", "Chemical reactions push electrons from one side to the other. Connect a "
     "circuit and those electrons flow through your device, doing work on the way."),
    ("why does ice float", "Water is weird: it's denser as a liquid than as a solid. Ice crystals lock into "
     "an open structure, so ice is about 9% less dense, which means it floats."),
    ("how does gps work", "Satellites broadcast the time constantly. Your phone notes how long each signal "
     "took to arrive and trilaterates its position from four or more of them. It needs no signal from you."),
    ("why do we get goosebumps", "A leftover from ancestors with more hair: cold or fear triggers tiny "
     "muscles that raise each hair. It still helps a little with warmth and makes you look bigger."),
    ("how does a microwave heat food", "Microwaves make water molecules rotate billions of times a second. "
     "That friction heats the water, and the water heats the rest of the food."),
    ("why is the sea level rising", "Warmer water expands, and glaciers and ice sheets are melting into the "
     "ocean. Both push the average sea level up, slowly but relentlessly."),
    ("how does an airplane fly", "Wings deflect air downward; Newton's third law pushes the plane up. The "
     "curved shape helps, but the main trick is angle and speed, not lift magic."),
    ("what causes rain", "Water evaporates, rises, cools, and condenses onto dust particles as droplets. "
     "When droplets collide into drops heavy enough, gravity wins and they fall."),
    ("why does my phone battery drain fast", "Usually screen brightness, background apps waking the radio, "
     "or an old battery with reduced capacity. Check battery usage by app — the culprit is nearly always at "
     "the top of that list."),
    ("how does a search engine work", "It crawls pages, stores what it finds in an index, then ranks results "
     "for your query using hundreds of signals like relevance, links, freshness and how people behave on the "
     "results."),
    ("what is inflation", "Prices across an economy creeping upward, which means each dollar buys a little "
     "less over time. Central banks aim for around 2% a year — low enough to be boring."),
    ("how do noise cancelling headphones work", "Microphones sample the incoming noise and the headphones "
     "play the exact opposite wave. Two opposite waves cancel where they meet your ear."),
    ("why do cats purr", "Often contentment, but also self-soothing when stressed or hurt. The vibration may "
     "even help healing — it's studied, not settled."),
    ("how does a hard drive store data", "Tiny regions of a magnetic disc get flipped one way or the other "
     "to represent bits. A read head detects the direction as the disc spins past."),
    ("what is entropy", "A measure of how many ways a system's parts can be arranged. Systems drift toward "
     "more arrangements, which is why messes appear and heat spreads out."),
    ("why is the moon sometimes visible in the day", "The moon is lit by the sun, not by the night. If it's "
     "above the horizon during daylight, you see it — often a pale daytime moon."),
    ("how does a vaccine work", "It shows your immune system a harmless piece or blueprint of a pathogen. "
     "Your body builds a response, so a real infection later gets met by an immune system that already knows "
     "the shape of the enemy."),
    ("what causes thunder", "Lightning heats the air to around 30,000°C in an instant. That air expands "
     "violently and the shockwave reaches your ears as thunder."),
    ("why does time feel faster as you age", "Novelty compresses less, and memory is denser when things are "
     "new. A year at 40 is a smaller fraction of your life than a year at 10, and fewer firsts make it blur."),
]
EXPLAIN_Q = ["why", "how does that work", "tell me more", "go on"]

def build_explanations():
    for topic, answer in EXPLAIN:
        C("explain", topic, answer)
        if maybe(0.55):
            C("explain", topic, answer, pick(["why", "how do you know that", "explain simpler", "more"]),
              pick([f"Short version: {answer.split('.')[0]}.",
                    f"Simplest way to hold it: {answer.split('.')[0]}.",
                    f"The core idea is the first sentence — {answer.split('.')[0]}."]))
        if maybe(0.2):
            C("explain", pick(["i'm bored", "tell me something interesting"]),
              pick([f"Okay, here's one: {answer}", f"Here's something: {answer}",
                    f"Random fact: {answer}"]))

# ----------------------------------------------------------------------------- 17. opinions (personality, not sycophancy)
OPINIONS = [
    ("is pineapple on pizza okay", "It's fine. Sweet and salty works elsewhere, so the outrage is mostly "
     "habit. I wouldn't order it, but I'd eat it without complaint."),
    ("what's the best programming language", "For getting things done quickly, Python. For the web, "
     "JavaScript. For learning how machines actually work, C. 'Best' always means best for a job."),
    ("tabs or spaces", "Spaces. One setting, no arguments, and diffs stop lying to you."),
    ("is a hot dog a sandwich", "Structurally yes, culturally no. These debates are fun because the "
     "definition is doing all the work, and the definition was never agreed on."),
    ("what's the worst trend in tech right now", "Putting a chatbot in everything. Most products need better "
     "search, not a personality."),
    ("should i learn rust or go", "Rust if you want to care about memory and correctness. Go if you want to "
     "ship a network service this week and not think about it again."),
    ("is it okay to use ai for homework", "For explaining and checking your reasoning, sure. Handing in "
     "generated work as your own is a different thing — it's dishonest, and you skip the part that builds "
     "your brain."),
    ("what do you think about social media", "It's an attention market. It's genuinely good at connection and "
     "genuinely bad at moderation, and the defaults are tuned for the second one."),
    ("best way to spend a rainy day", "Something with your hands and no screen: cook, fix, draw, sort out one "
     "messy drawer. Rain is good permission to be slow."),
    ("is coffee good for you", "Net positive for most people at moderate amounts — around 3-4 cups. Sleep and "
     "anxiety are the two places it bites."),
    ("what's overrated", "Hustle culture. Most '10x' productivity advice is just unpaid overtime with better "
     "branding."),
    ("what's underrated", "Boring consistency. Sleep, walking, and showing up slightly early beat almost every "
     "clever system."),
    ("should i get a dog or a cat", "Dog if you want a companion that needs a schedule. Cat if you want a "
     "housemate with boundaries. Honestly: dog if you'll walk it daily, cat if you won't."),
    ("is 6pm too early for dinner", "No. The 8pm dinner crowd is just hungry and pretending."),
    ("what's the best way to learn something new", "Build something small with it within the first day. "
     "The first day is where curiosity either sticks or evaporates."),
    ("do you think ai will take jobs", "It'll take tasks first and jobs second, and the people who get hit "
     "hardest are usually those with the least room to retrain. That's a policy problem as much as a "
     "technology one."),
    ("is it weird to eat cereal for dinner", "No. It's food, it's fast, and the only judge is a marketing "
     "department from 1950."),
    ("what music should i listen to while coding", "Instrumental, or anything you've heard enough times that "
     "it stops being a story. Lyrics compete with the words in your head."),
    ("should i go to university", "If the field needs credentials or labs, yes. If it's something you can "
     "build a portfolio in, the portfolio often wins. Both are fine — just don't drift in by default."),
    ("what's the point of art", "It's how humans export feelings to each other. That's not decorative; it's "
     "one of the main ways we understand each other."),
]

def build_opinions():
    for q, a in OPINIONS:
        C("opinion", q, a)
        if maybe(0.5):
            C("opinion", q, a, pick(["why", "you're wrong", "convince me", "hm"]),
              pick(["Because it's the version that survives contact with real life.",
                    "I might be wrong — but that's my read, and I'd rather say it plainly.",
                    "Happy to be argued out of it. What's your take?"]))

# ----------------------------------------------------------------------------- 18. tech chat
TECH = [
    ("why is my laptop slow", "Usually one of three things: the disk is full, too many things start at "
     "boot, or the CPU is throttling because of heat or a dying battery. Check disk space first, then "
     "startup items, then whether the fan sounds like a jet."),
    ("what's the difference between ram and storage", "RAM is the desk you're working on — fast and wiped "
     "when you leave. Storage is the filing cabinet — slower, but everything stays. Not enough RAM and "
     "things crawl; not enough storage and nothing fits."),
    ("what's a vps", "A virtual private server: a slice of a physical machine you rent, with root access, "
     "running all the time. It's the simplest way to have something always online."),
    ("how do i make my website faster", "Compress images, ship less JavaScript, cache what doesn't change, "
     "and measure before and after. Most slow sites are 90% images and unused scripts."),
    ("what's an api key", "A secret string that identifies your app when it calls a service. Treat it like "
     "a password — never in client-side code, never in a public repo."),
    ("why does my code work locally but not in production", "Classic causes: different environment "
     "variables, a dependency version that resolved differently, file paths that only exist on your "
     "machine, or a case-sensitive filesystem. Log the environment, not just the error."),
    ("what's docker", "A way to package an app with its exact environment so it runs the same everywhere. "
     "It bundles libraries and settings into an image you can run as a container."),
    ("what is git actually doing", "It stores snapshots of your files and lets you move between them. "
     "Branches are just labels pointing at a snapshot, which is why branching in git is instant and cheap."),
    ("how do i back up my files properly", "Three copies: the one you use, one on a different device, one "
     "off-site. If it's all in the same house it's one bad day from gone."),
    ("what's the cloud", "Someone else's computer, rented by the minute, with better uptime than yours. "
     "The magic is elasticity, not airiness."),
    ("why do websites ask about cookies", "Cookies store small bits of state — logins, preferences, tracking "
     "IDs. Laws require asking because some of them follow you across sites."),
    ("how do i know if a website is safe", "Check the domain carefully, look for https, and be suspicious of "
     "urgency. If a page rushes you, close it and go to the site directly."),
    ("what's a hash function", "A function that turns any input into a fixed-size fingerprint. Same input, "
     "same output; tiny change, totally different output; and you can't reverse it."),
    ("should i use a password manager", "Yes. One strong master password and unique random passwords "
     "everywhere else is strictly better than reusing variations of your dog's name."),
    ("what's a neural network", "Layers of numbers multiplied by learned weights, with nonlinearities "
     "between them. Training nudges the weights so the output gets less wrong. That's the whole trick, "
     "scaled up."),
]

def build_tech():
    for q, a in TECH:
        C("tech", q, a)
        if maybe(0.45):
            C("tech", q, a, pick(["thanks", "makes sense", "ok and how do i start"]),
              pick(["Start with the smallest version and measure — most of this is diagnosis, not magic.",
                    "Start with the free option. Upgrade when it hurts.",
                    "Pick one thing from that list and do it today; the rest can wait."]))

# ----------------------------------------------------------------------------- 19. creative + rewrites
HAIKU = [
    ("rain", "Rain on the window —\nthe whole city turns quiet,\nthinking to itself."),
    ("the sea", "The sea keeps coming —\nno memory of last night,\nno promise either."),
    ("winter", "Cold morning light comes\nand finds the kettle steaming,\npatient as a friend."),
    ("city", "Ten thousand windows\neach with somebody awake,\nnone of them looking."),
    ("old laptop", "Old laptop whirs on —\nits fan a tiny weather,\nwarm in the evening."),
    ("moon", "The moon doesn't care\nthat we named it, or that we\nkeep watching it."),
]
def build_creative():
    for topic, poem in HAIKU:
        C("creative", pick([f"write a haiku about {topic}", f"haiku about {topic}?",
                            f"give me a haiku on {topic}"]), poem)
    for _ in range(120):
        C("creative", pick(["write me a one-sentence story", "one sentence story please",
                            "tell me a very short story"]),
          pick(["The robot learned to say 'good morning' and then had nothing else to do with it.",
                "She planted the last seed on the roof and waited for the city to notice.",
                "The map was accurate for every street except the one he needed.",
                "It rained inside the museum and nobody wanted to be the one to mention it.",
                "The lighthouse blinked twice, which wasn't in the manual, and the ship turned.",
                "He sold his guitar to buy a ticket, then found the concert was free."]))
    REWRITE = [
        ("i can't come to the meeting", "formal", "Unfortunately, I will be unable to attend the meeting."),
        ("i can't come to the meeting", "casual", "Yeah, I can't make the meeting — sorry!"),
        ("send me the file", "formal", "Could you please send me the file?"),
        ("send me the file", "casual", "Hey, can you send me that file when you get a sec?"),
        ("this is wrong", "polite", "I think there might be an issue here — could we double-check it?"),
        ("this is wrong", "blunt", "This is wrong."),
        ("i don't know", "confident", "I don't have that answer yet — I'll find out and come back to you."),
    ]
    for base, tone, out in REWRITE:
        C("creative", pick([f"say '{base}' but {tone}", f"rewrite '{base}' to sound {tone}",
                            f"make this more {tone}: {base}"]), out)

# ----------------------------------------------------------------------------- 14. assembly
SYSTEM_PROMPTS = [
    ("You are v0idgpt, a small assistant running locally in the browser. Answer what you're asked, "
     "be concise, and admit uncertainty.", 0.82),
    ("You are v0idgpt, a 135M parameter assistant running in the user's browser. Be direct, warm and "
     "honest about your limits.", 0.08),
    ("You are v0idgpt. Keep replies short and useful. Say when you don't know.", 0.06),
    ("", 0.04),
]

def choose_system():
    r = RNG.random()
    acc = 0
    for text, w in SYSTEM_PROMPTS:
        acc += w
        if r <= acc:
            return text
    return SYSTEM_PROMPTS[0][0]

def token_estimate(text):
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("building corpus ...")
    build_arithmetic(2400)
    build_word_math(800)
    for _ in range(4):         # each pass samples different paraphrases; de-dup keeps it honest
        build_knowledge()
    build_topic_drills(1300)
    build_corrections(900)
    build_memory(800)
    build_reasoning(1000)
    build_style(800)
    build_coding(500)
    build_advice(500)
    build_ideas(400)
    build_safety()
    build_language(0)
    build_explanations()
    build_opinions()
    build_tech()
    build_creative()
    generated = len(CONVS)
    hw = load_handwritten()
    hw_convs = list(CONVS[generated:])
    for c in hw_convs:          # up-weight the hand-written voice ~2x
        CONVS.append({"category": c["category"], "messages": c["messages"]})
    print(f"  generated {generated}, hand-written {hw} (x2 = {len(hw_convs) * 2} in corpus)")

    # ---- de-duplicate identical conversations, keep first occurrence
    seen, uniq = set(), []
    for c in CONVS:
        key = tuple((m["role"], m["content"]) for m in c["messages"])
        if key in seen:
            continue
        if not c["messages"] or c["messages"][-1]["role"] != "assistant":
            continue
        seen.add(key)
        uniq.append(c)
    RNG.shuffle(uniq)
    for c in uniq:
        c["system"] = choose_system()
    print(f"  unique conversations: {len(uniq)} (dropped {len(CONVS) - len(uniq)})")

    # ---- write
    jl = os.path.join(OUT_DIR, "dataset.jsonl")
    with open(jl, "w", encoding="utf-8") as f:
        for c in uniq:
            f.write(json.dumps({"category": c["category"], "system": c["system"],
                                "messages": c["messages"]}, ensure_ascii=False) + "\n")
    tx = os.path.join(OUT_DIR, "dataset.txt")
    lines = 0
    with open(tx, "w", encoding="utf-8") as f:
        for c in uniq:
            for m in c["messages"]:
                f.write(f"{'user' if m['role'] == 'user' else 'assistant'}: {m['content']}\n")
                if m["content"].count("\n"):
                    f.write("\n")
                lines += 1
            f.write("\n")
    cats, ntok, nassist, nturns = {}, 0, 0, 0
    for c in uniq:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
        ntok += token_estimate(c["system"])
        for m in c["messages"]:
            t = token_estimate(m["content"])
            ntok += t + 3
            nturns += 1
            if m["role"] == "assistant":
                nassist += t
    stats = {
        "conversations": len(uniq), "dialogue_lines": lines, "turns": nturns,
        "approx_tokens": ntok, "approx_assistant_tokens": nassist,
        "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "seed": SEED,
    }
    json.dump(stats, open(os.path.join(OUT_DIR, "dataset-stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))
    print(f"  wrote {jl} ({os.path.getsize(jl)/1e6:.2f} MB) and {tx} ({os.path.getsize(tx)/1e6:.2f} MB)")
    check_arithmetic(uniq)
    return uniq

# ----------------------------------------------------------------------------- 15. checks
ARITH_Q = [
    (re.compile(r"^what(?:'s| is)\s+(\d+)\s*\+\s*(\d+)\s*\??$", re.I), lambda a, b: a + b),
    (re.compile(r"^what(?:'s| is)\s+(\d+)\s*-\s*(\d+)\s*\??$", re.I), lambda a, b: a - b),
    (re.compile(r"^what(?:'s| is)\s+(\d+)\s*(?:times|x|\*|×)\s*(\d+)\s*\??$", re.I), lambda a, b: a * b),
    (re.compile(r"^(\d+)\s*\+\s*(\d+)\s*=?\s*\??$"), lambda a, b: a + b),
    (re.compile(r"^(\d+)\s*(?:times|x|\*|×)\s*(\d+)\s*=?\s*\??$", re.I), lambda a, b: a * b),
    (re.compile(r"^what(?:'s| is)\s+(\d+)\s+plus\s+(\d+)\s*\??$", re.I), lambda a, b: a + b),
    (re.compile(r"^what(?:'s| is)\s+(\d+)\s+minus\s+(\d+)\s*\??$", re.I), lambda a, b: a - b),
]

def check_arithmetic(convs):
    bad, checked = [], 0
    for c in convs:
        u = c["messages"][0]["content"]
        a = c["messages"][1]["content"] if len(c["messages"]) > 1 else ""
        for rx, fn in ARITH_Q:
            m = rx.match(u.strip())
            if not m:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            expect = fn(x, y)
            eq = re.search(r"=\s*(-?\d[\d,]*)", a)
            src = eq.group(1) if eq else None
            if src is None:
                first = re.search(r"-?\d[\d,]*", a)
                if not first:
                    break
                src = first.group(0)
            got = int(src.replace(",", ""))
            checked += 1
            if got != expect:
                bad.append((u, a[:90], expect, got))
            break
    print(f"  arithmetic check: {checked} verifiable answers, {len(bad)} mismatches")
    for row in bad[:12]:
        print("   MISMATCH", row)
    return bad

if __name__ == "__main__":
    main()
