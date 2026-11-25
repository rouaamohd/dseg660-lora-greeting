"""Base questions for the training and development sets.

Every base question has one correct, concise answer.  Answers for computed tasks
(units, arithmetic, word problems, counting, sorting, calendar, JSON) are produced
programmatically; knowledge tasks come from hand-checked lists.  The 16 evaluation
questions (and anything asking the same thing) are excluded in make_data.py.

Each question dict has:
  task, question, answer (text used in the training reply), check (scoring spec),
  details (content restrictions that do not change the answer).
"""
import json
import random
from fractions import Fraction

# ---------------------------------------------------------------- helpers

def fmt_num(x):
    x = Fraction(x)
    if x.denominator == 1:
        return str(x.numerator)
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    return s


def text_q(task, question, answer, accepted=(), details=None):
    return dict(task=task, question=question, answer=answer,
                check=dict(type="text", expected=answer, accepted=list(accepted)),
                details=details)


def num_q(task, question, answer, value, accepted=(), details=None):
    return dict(task=task, question=question, answer=answer,
                check=dict(type="number", expected=answer, value=str(value), accepted=list(accepted)),
                details=details)


# ---------------------------------------------------------------- per-task detail restrictions
# Content restrictions ("omit details") that leave the answer unchanged.
# train / dev lists are disjoint so the dev set tests unseen restriction wording.
DETAILS = {
    "units": (["the working", "the conversion factor", "the conversion steps", "any explanation of the math", "the formula you used"],
              ["the multiplication you did", "a breakdown of the conversion"]),
    "arithmetic": (["the working", "the intermediate steps", "any explanation", "the long multiplication", "a step-by-step breakdown"],
                   ["the partial products", "how you got the result"]),
    "capital": (["facts about the population", "the history of the city", "travel tips", "any extra details about the country", "background on the region"],
                ["landmarks worth visiting", "information about the language spoken there"]),
    "element": (["the atomic mass", "the element's properties", "its group in the periodic table", "the discovery history", "uses of the element"],
                ["the electron configuration", "where the element is found"]),
    "science": (["the history behind it", "the derivation", "any equations", "worked examples", "extra background"],
                ["who it is named after", "the unit's abbreviation history"]),
    "plural": (["an example sentence", "the grammar rule", "the etymology", "notes on usage", "the Latin origin"],
               ["a definition of the word", "other irregular plurals"]),
    "spanish": (["the pronunciation guide", "grammar notes", "alternative translations", "a word-by-word breakdown", "notes on regional variation"],
                ["the literal meaning", "an explanation of the verb tense"]),
    "french": (["the pronunciation guide", "grammar notes", "alternative translations", "a word-by-word breakdown", "notes on gender agreement"],
               ["the literal meaning", "an explanation of the word order"]),
    "python": (["a Markdown code block", "an explanation of how it works", "example output", "comments", "a full program"],
               ["a description of slicing rules", "backticks around the code"]),
    "json": (["an explanation", "a code fence", "comments", "a description of the schema", "any extra fields"],
             ["a note about data types", "pretty-printing"]),
    "extraction": (["the rest of the sentence", "an explanation", "the other details in the sentence", "a quote of the full text", "your reasoning"],
                   ["any other numbers in the text", "a label for the field"]),
    "calendar": (["the counting steps", "the list of months in between", "the explanation", "the calculation", "a calendar layout"],
                 ["how you counted", "the days in each month"]),
    "sentiment": (["the explanation of your choice", "a confidence score", "your reasoning", "a summary of the text", "the other labels"],
                  ["why you picked that label", "a rating out of ten"]),
    "word_problem": (["the steps of the calculation", "the working", "the equation", "any explanation", "a restatement of the problem"],
                     ["the intermediate totals", "how you set up the problem"]),
    "count": (["a letter-by-letter breakdown", "the spelling", "the working", "an explanation", "a list of the items you counted"],
              ["the counting process", "each letter on its own"]),
    "sorting": (["the reasoning", "definitions of the words", "an explanation of the order", "any numbering", "extra commentary"],
                ["why that order is correct", "a description of each item"]),
    "general": (["any explanation", "background details", "extra facts", "historical context", "the reasoning"],
                ["related trivia", "a longer description"]),
}

# ---------------------------------------------------------------- units
UNIT_NAMES = {
    "hours": "hour", "minutes": "minute", "seconds": "second", "days": "day", "weeks": "week",
    "kilometers": "kilometer", "meters": "meter", "centimeters": "centimeter", "millimeters": "millimeter",
    "kilograms": "kilogram", "grams": "gram", "milligrams": "milligram", "liters": "liter", "milliliters": "milliliter",
    "feet": "foot", "inches": "inch", "yards": "yard",
}
UNIT_PAIRS = [
    ("hours", "minutes", Fraction(60), ["1.5", "2.5", "0.75", "0.2", "3.25", "1.1", "0.6", "4.5", "0.35", "2.2"]),
    ("minutes", "seconds", Fraction(60), ["2.5", "4", "0.75", "12", "1.25", "7", "0.4"]),
    ("days", "hours", Fraction(24), ["3", "1.5", "0.25", "7", "2.5", "10"]),
    ("weeks", "days", Fraction(7), ["3", "6", "2.5", "12", "9"]),
    ("kilometers", "meters", Fraction(1000), ["2.4", "0.65", "3.75", "12", "0.08"]),
    ("meters", "centimeters", Fraction(100), ["1.85", "0.42", "3.6", "7"]),
    ("centimeters", "millimeters", Fraction(10), ["4.5", "12", "0.8", "36"]),
    ("kilograms", "grams", Fraction(1000), ["2.3", "0.75", "1.05", "6"]),
    ("grams", "milligrams", Fraction(1000), ["0.5", "2.25", "0.04"]),
    ("liters", "milliliters", Fraction(1000), ["1.5", "0.33", "2.75", "0.06"]),
    ("feet", "inches", Fraction(12), ["3", "5", "2.5", "11"]),
    ("yards", "feet", Fraction(3), ["4", "12", "2.5", "20"]),
    ("minutes", "hours", Fraction(1, 60), ["90", "150", "45", "210", "30", "75"]),
    ("meters", "kilometers", Fraction(1, 1000), ["2500", "750", "4200", "125"]),
    ("grams", "kilograms", Fraction(1, 1000), ["3500", "250", "1200"]),
    ("milliliters", "liters", Fraction(1, 1000), ["1500", "250", "4000"]),
    ("seconds", "minutes", Fraction(1, 60), ["180", "90", "300", "45"]),
    ("hours", "days", Fraction(1, 24), ["48", "36", "120"]),
]
UNIT_TEMPLATES = [
    "Convert {v} {f} to {t}.",
    "How many {t} are in {v} {f}?",
    "Express {v} {f} in {t}.",
    "What is {v} {f} in {t}?",
    "Change {v} {f} into {t}.",
    "{v} {f} equals how many {t}?",
]


def units_questions(rng):
    out = []
    for f, t, factor, vals in UNIT_PAIRS:
        for v in vals:
            if f == "hours" and t == "minutes" and v == "0.45":
                continue  # evaluation question
            res = Fraction(v) * factor
            rs = fmt_num(res)
            unit = UNIT_NAMES[t] if res == 1 else t
            fu = UNIT_NAMES[f] if Fraction(v) == 1 else f
            q = rng.choice(UNIT_TEMPLATES).format(v=v, f=fu, t=t)
            out.append(num_q("units", q, f"{rs} {unit}", rs))
    return out


# ---------------------------------------------------------------- arithmetic
def arithmetic_questions(rng):
    out = []
    mult_t = ["What is {a} multiplied by {b}?", "What is {a} times {b}?", "Compute {a} × {b}.", "Calculate {a} * {b}.", "What do you get when you multiply {a} by {b}?", "Find the product of {a} and {b}."]
    add_t = ["What is {a} plus {b}?", "Add {a} and {b}.", "Compute {a} + {b}.", "What is the sum of {a} and {b}?"]
    sub_t = ["What is {a} minus {b}?", "Subtract {b} from {a}.", "Compute {a} - {b}.", "What is {a} take away {b}?"]
    div_t = ["What is {a} divided by {b}?", "Divide {a} by {b}.", "Compute {a} ÷ {b}.", "What is {a} / {b}?"]
    sq_t = ["What is {a} squared?", "What is the square of {a}?", "Compute {a}²."]
    seen = set()
    while len(out) < 60:
        kind = rng.choice(["mul", "mul", "add", "sub", "div", "sq"])
        if kind == "mul":
            a, b = rng.randint(11, 99), rng.randint(2, 12)
            if {a, b} == {31, 4}:
                continue
            if rng.random() < 0.5:
                a, b = b, a
            res, q = a * b, rng.choice(mult_t).format(a=a, b=b)
        elif kind == "add":
            a, b = rng.randint(12, 899), rng.randint(12, 899)
            res, q = a + b, rng.choice(add_t).format(a=a, b=b)
        elif kind == "sub":
            a = rng.randint(50, 999); b = rng.randint(10, a - 1)
            res, q = a - b, rng.choice(sub_t).format(a=a, b=b)
        elif kind == "div":
            b = rng.randint(2, 12); res = rng.randint(3, 60); a = b * res
            q = rng.choice(div_t).format(a=a, b=b)
        else:
            a = rng.randint(3, 25); res = a * a
            q = rng.choice(sq_t).format(a=a)
        if res == 124 or q in seen:
            continue
        seen.add(q)
        out.append(num_q("arithmetic", q, str(res), res))
    return out


# ---------------------------------------------------------------- capitals
CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
    ("China", "Beijing"), ("India", "New Delhi"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
    ("Brazil", "Brasília", ["Brasilia"]), ("Argentina", "Buenos Aires"), ("Mexico", "Mexico City"), ("Egypt", "Cairo"),
    ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Morocco", "Rabat"), ("Ethiopia", "Addis Ababa"),
    ("Russia", "Moscow"), ("Ukraine", "Kyiv", ["Kiev"]), ("Poland", "Warsaw"), ("Sweden", "Stockholm"), ("Norway", "Oslo"),
    ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Iceland", "Reykjavik", ["Reykjavík"]), ("Ireland", "Dublin"),
    ("Greece", "Athens"), ("Turkey", "Ankara"), ("Hungary", "Budapest"), ("Austria", "Vienna"), ("Switzerland", "Bern", ["Berne"]),
    ("Belgium", "Brussels"), ("the Czech Republic", "Prague"), ("Slovakia", "Bratislava"), ("Romania", "Bucharest"),
    ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Slovenia", "Ljubljana"), ("Latvia", "Riga"),
    ("Lithuania", "Vilnius"), ("Peru", "Lima"), ("Chile", "Santiago"), ("Colombia", "Bogotá", ["Bogota"]), ("Venezuela", "Caracas"),
    ("Ecuador", "Quito"), ("Uruguay", "Montevideo"), ("Paraguay", "Asunción", ["Asuncion"]), ("Cuba", "Havana"), ("Jamaica", "Kingston"),
    ("South Korea", "Seoul"), ("Vietnam", "Hanoi"), ("Thailand", "Bangkok"), ("the Philippines", "Manila"),
    ("Malaysia", "Kuala Lumpur"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"),
    ("Afghanistan", "Kabul"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"), ("Qatar", "Doha"),
    ("the United Arab Emirates", "Abu Dhabi"), ("Jordan", "Amman"), ("Lebanon", "Beirut"), ("Syria", "Damascus"), ("Oman", "Muscat"),
    ("Mongolia", "Ulaanbaatar", ["Ulan Bator"]), ("Kazakhstan", "Astana"), ("New Zealand", "Wellington"), ("Tunisia", "Tunis"),
    ("Algeria", "Algiers"), ("Senegal", "Dakar"), ("Uganda", "Kampala"), ("Zimbabwe", "Harare"),
    ("Zambia", "Lusaka"), ("Angola", "Luanda"), ("the Netherlands", "Amsterdam"), ("Malta", "Valletta"), ("Cyprus", "Nicosia"),
    ("Albania", "Tirana"), ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Belarus", "Minsk"),
    ("Panama", "Panama City"), ("Costa Rica", "San José", ["San Jose"]), ("Nicaragua", "Managua"), ("Haiti", "Port-au-Prince"),
    ("Madagascar", "Antananarivo"), ("Rwanda", "Kigali"), ("Mali", "Bamako"), ("Laos", "Vientiane"), ("Cambodia", "Phnom Penh"),
    ("the United Kingdom", "London"), ("Uzbekistan", "Tashkent"), ("Kyrgyzstan", "Bishkek"), ("Bahrain", "Manama"),
    ("Portugal", "Lisbon"), ("Kuwait", "Kuwait City"), ("Libya", "Tripoli"), ("Sudan", "Khartoum"), ("Fiji", "Suva"),
]
CAPITAL_T = ["What is the capital of {c}?", "Name the capital of {c}.", "Which city is the capital of {c}?",
             "Tell me the capital city of {c}.", "What city serves as the capital of {c}?", "Identify the capital of {c}."]


def capital_questions(rng):
    out = []
    for item in CAPITALS:
        c, cap = item[0], item[1]
        acc = item[2] if len(item) > 2 else []
        q = rng.choice(CAPITAL_T).format(c=c)
        out.append(text_q("capital", q, cap, acc))
    return out


# ---------------------------------------------------------------- elements
ELEMENTS = [  # symbol, name, atomic number, accepted names
    ("H", "Hydrogen", 1, []), ("He", "Helium", 2, []), ("Li", "Lithium", 3, []), ("Be", "Beryllium", 4, []),
    ("B", "Boron", 5, []), ("C", "Carbon", 6, []), ("N", "Nitrogen", 7, []), ("O", "Oxygen", 8, []), ("F", "Fluorine", 9, []),
    ("Ne", "Neon", 10, []), ("Na", "Sodium", 11, []), ("Mg", "Magnesium", 12, []), ("Al", "Aluminium", 13, ["Aluminum"]),
    ("Si", "Silicon", 14, []), ("P", "Phosphorus", 15, []), ("S", "Sulfur", 16, ["Sulphur"]), ("Cl", "Chlorine", 17, []),
    ("Ar", "Argon", 18, []), ("K", "Potassium", 19, []), ("Ca", "Calcium", 20, []), ("Ti", "Titanium", 22, []),
    ("Cr", "Chromium", 24, []), ("Mn", "Manganese", 25, []), ("Fe", "Iron", 26, []), ("Co", "Cobalt", 27, []),
    ("Ni", "Nickel", 28, []), ("Cu", "Copper", 29, []), ("Zn", "Zinc", 30, []), ("Br", "Bromine", 35, []),
    ("Kr", "Krypton", 36, []), ("Ag", "Silver", 47, []), ("Sn", "Tin", 50, []), ("I", "Iodine", 53, []),
    ("Xe", "Xenon", 54, []), ("Ba", "Barium", 56, []), ("W", "Tungsten", 74, []), ("Pt", "Platinum", 78, []),
    ("Au", "Gold", 79, []), ("Hg", "Mercury", 80, []), ("Pb", "Lead", 82, []), ("U", "Uranium", 92, []),
    ("Sb", "Antimony", 51, []), ("Bi", "Bismuth", 83, []), ("Cd", "Cadmium", 48, []), ("Mo", "Molybdenum", 42, []),
    ("V", "Vanadium", 23, []), ("Se", "Selenium", 34, []), ("As", "Arsenic", 33, []), ("Ra", "Radium", 88, []),
    ("Pu", "Plutonium", 94, []), ("Ga", "Gallium", 31, []), ("Ge", "Germanium", 32, []),
]
SYM_T = ["Which element has the chemical symbol {s}?", "What element does the symbol {s} stand for?",
         "Name the element with the symbol {s}.", "The chemical symbol {s} belongs to which element?",
         "Which element is represented by {s} on the periodic table?"]
NAME_T = ["What is the chemical symbol for {n}?", "Give the chemical symbol of {n}.", "Which symbol stands for {n}?"]
NUM_T = ["Which element has atomic number {z}?", "Name the element whose atomic number is {z}."]


def element_questions(rng):
    out = []
    for s, n, z, acc in ELEMENTS:
        kind = rng.choice(["sym", "sym", "name", "num"])
        if kind == "sym":
            out.append(text_q("element", rng.choice(SYM_T).format(s=s), n, acc))
        elif kind == "name":
            q = dict(task="element", question=rng.choice(NAME_T).format(n=n.lower()), answer=s,
                     check=dict(type="symbol", expected=s, accepted=[]), details=None)
            out.append(q)
        else:
            out.append(text_q("element", rng.choice(NUM_T).format(z=z), n, acc))
    return out


# ---------------------------------------------------------------- science
SI_UNITS = [
    ("force", "Newton", ["N"]), ("energy", "Joule", ["J"]), ("power", "Watt", ["W"]), ("pressure", "Pascal", ["Pa"]),
    ("electric charge", "Coulomb", ["C"]), ("electric potential difference", "Volt", ["V"]),
    ("electrical resistance", "Ohm", []), ("capacitance", "Farad", ["F"]), ("frequency", "Hertz", ["Hz"]),
    ("electric current", "Ampere", ["amp"]), ("thermodynamic temperature", "Kelvin", ["K"]),
    ("amount of substance", "Mole", ["mol"]), ("luminous intensity", "Candela", ["cd"]), ("mass", "Kilogram", ["kg"]),
    ("length", "Metre", ["Meter", "m"]), ("time", "Second", ["s"]), ("radioactivity", "Becquerel", ["Bq"]),
    ("absorbed radiation dose", "Gray", ["Gy"]), ("electrical conductance", "Siemens", ["S"]), ("illuminance", "Lux", ["lx"]),
]
SI_T = ["What is the SI unit of {q}?", "Which SI unit is used to measure {q}?", "Name the SI unit for {q}.",
        "In SI units, what is {q} measured in?"]
SCIENCE_FACTS = [
    ("What gas do plants absorb from the air for photosynthesis?", "Carbon dioxide", ["CO2"]),
    ("What is the chemical formula of water?", "H2O", ["H₂O"]),
    ("Which planet is closest to the Sun?", "Mercury", []),
    ("Which part of the cell is known as its powerhouse?", "The mitochondria", ["mitochondrion", "mitochondria"]),
    ("Which subatomic particle carries a negative charge?", "The electron", ["electron"]),
    ("What force pulls objects toward the center of the Earth?", "Gravity", []),
    ("What is the speed of light in a vacuum, approximately, in kilometers per second?", "About 300,000 km/s", ["300,000", "299,792", "300000"]),
    ("Which organ in the human body produces insulin?", "The pancreas", ["pancreas"]),
    ("What is the pH of pure water at 25°C?", "7", []),
    ("What type of bond involves sharing electron pairs between atoms?", "A covalent bond", ["covalent"]),
    ("What is the most abundant gas in Earth's atmosphere?", "Nitrogen", []),
    ("Which planet is known as the Red Planet?", "Mars", []),
    ("What is the center of an atom called?", "The nucleus", ["nucleus"]),
    ("Which blood cells carry oxygen around the body?", "Red blood cells", ["erythrocytes"]),
    ("What is the boiling point of water at sea level in degrees Celsius?", "100°C", ["100 °C", "100 degrees"]),
    ("What process turns liquid water into vapor?", "Evaporation", ["vaporization", "boiling"]),
    ("Which vitamin does the skin make when exposed to sunlight?", "Vitamin D", []),
    ("What is the hardest natural mineral?", "Diamond", []),
    ("What is the largest planet in the solar system?", "Jupiter", []),
    ("How many chromosomes do humans usually have?", "46", ["23 pairs"]),
]


def science_questions(rng):
    out = []
    for q, unit, acc in SI_UNITS:
        out.append(text_q("science", rng.choice(SI_T).format(q=q), unit, acc))
    for q, a, acc in SCIENCE_FACTS:
        out.append(text_q("science", q, a, acc))
    return out


# ---------------------------------------------------------------- plurals
PLURALS = [("mouse", "Mice"), ("child", "Children"), ("cactus", "Cacti"), ("criterion", "Criteria"),
           ("phenomenon", "Phenomena"), ("analysis", "Analyses"), ("thesis", "Theses"), ("goose", "Geese"),
           ("tooth", "Teeth"), ("foot", "Feet"), ("ox", "Oxen"), ("curriculum", "Curricula"), ("fungus", "Fungi"),
           ("nucleus", "Nuclei"), ("radius", "Radii"), ("alumnus", "Alumni"), ("crisis", "Crises"),
           ("vertex", "Vertices"), ("larva", "Larvae"), ("knife", "Knives"), ("leaf", "Leaves"), ("wolf", "Wolves"),
           ("woman", "Women"), ("person", "People"), ("sheep", "Sheep"), ("deer", "Deer"), ("louse", "Lice"),
           ("stimulus", "Stimuli"), ("hypothesis", "Hypotheses"), ("diagnosis", "Diagnoses"), ("oasis", "Oases"),
           ("axis", "Axes"), ("half", "Halves"), ("life", "Lives"), ("potato", "Potatoes"), ("tomato", "Tomatoes"),
           ("hero", "Heroes"), ("echo", "Echoes"), ("millennium", "Millennia"), ("stratum", "Strata"),
           ("matrix", "Matrices"), ("appendix", "Appendices"), ("parenthesis", "Parentheses"), ("thief", "Thieves"),
           ("loaf", "Loaves"), ("man", "Men"), ("die", "Dice"), ("basis", "Bases"), ("ellipsis", "Ellipses"),
           ("nebula", "Nebulae"), ("vertebra", "Vertebrae"), ("syllabus", "Syllabi")]
PLURAL_T = ["Give the plural of '{w}'.", "What is the plural of '{w}'?", "What's the plural form of the word '{w}'?",
            "Write the plural of '{w}'.", "How do you write '{w}' in the plural?", "Form the plural of the noun '{w}'."]
PLURAL_ACC = {"Syllabi": ["Syllabuses"], "Appendices": ["Appendixes"], "Matrices": ["Matrixes"], "Nebulae": ["Nebulas"],
              "Vertices": ["Vertexes"], "Cacti": ["Cactuses"], "Fungi": ["Funguses"]}


def plural_questions(rng):
    return [text_q("plural", rng.choice(PLURAL_T).format(w=w), p, PLURAL_ACC.get(p, [])) for w, p in PLURALS]


# ---------------------------------------------------------------- translations
SPANISH = [
    ("Good morning", "Buenos días", []), ("Thank you very much", "Muchas gracias", []),
    ("Where is the library?", "¿Dónde está la biblioteca?", []), ("I am hungry", "Tengo hambre", []),
    ("The house is big", "La casa es grande", []), ("I like coffee", "Me gusta el café", []),
    ("We are friends", "Somos amigos", []), ("The book is on the table", "El libro está en la mesa", []),
    ("What time is it?", "¿Qué hora es?", []), ("The cat is sleeping", "El gato está durmiendo", ["El gato duerme"]),
    ("I have two brothers", "Tengo dos hermanos", []), ("The water is cold", "El agua está fría", []),
    ("She reads a book", "Ella lee un libro", []), ("Happy birthday", "Feliz cumpleaños", []),
    ("See you tomorrow", "Hasta mañana", ["Nos vemos mañana"]), ("I speak Spanish", "Hablo español", ["Yo hablo español"]),
    ("The sky is blue", "El cielo es azul", []), ("My name is Ana", "Me llamo Ana", ["Mi nombre es Ana"]),
    ("The red car", "El coche rojo", ["El carro rojo", "El auto rojo"]), ("I live in Madrid", "Vivo en Madrid", []),
    ("It is raining", "Está lloviendo", ["Llueve"]), ("The children play in the park", "Los niños juegan en el parque", []),
    ("I need help", "Necesito ayuda", []), ("The door is open", "La puerta está abierta", []),
    ("Good night", "Buenas noches", []), ("How are you?", "¿Cómo estás?", ["¿Cómo está usted?"]),
    ("The food is delicious", "La comida está deliciosa", ["La comida es deliciosa"]),
    ("Where is the station?", "¿Dónde está la estación?", []), ("I want water", "Quiero agua", []),
    ("The green apple", "La manzana verde", []), ("We eat bread", "Comemos pan", []),
    ("The store opens at nine", "La tienda abre a las nueve", []), ("My mother cooks", "Mi madre cocina", ["Mi mamá cocina"]),
    ("The window is closed", "La ventana está cerrada", []), ("I don't understand", "No entiendo", []),
    ("The bird sings", "El pájaro canta", []), ("I am a student", "Soy estudiante", []),
    ("Close the door", "Cierra la puerta", ["Cierre la puerta"]), ("I like to read", "Me gusta leer", []),
    ("The tea is hot", "El té está caliente", []), ("Today is Monday", "Hoy es lunes", []),
    ("The white shirt", "La camisa blanca", []), ("We are going to the market", "Vamos al mercado", []),
    ("The flowers are yellow", "Las flores son amarillas", []), ("I have a question", "Tengo una pregunta", []),
    ("The sun is shining", "El sol brilla", ["Hace sol", "El sol está brillando"]),
]
SPANISH_T = ["Translate '{e}' into Spanish.", "How do you say '{e}' in Spanish?", "What is '{e}' in Spanish?",
             "Give the Spanish for '{e}'.", "Render '{e}' in Spanish.", "Put '{e}' into Spanish."]
FRENCH = [
    ("Good evening", "Bonsoir", []), ("Thank you", "Merci", []), ("The black cat", "Le chat noir", []),
    ("I am happy", "Je suis heureux", ["Je suis heureuse", "Je suis content", "Je suis contente"]),
    ("Where is the station?", "Où est la gare ?", ["Où est la gare?"]), ("I like apples", "J'aime les pommes", []),
    ("The house is white", "La maison est blanche", []), ("We are students", "Nous sommes étudiants", ["Nous sommes des étudiants"]),
    ("The big tree", "Le grand arbre", []), ("See you soon", "À bientôt", []), ("I speak French", "Je parle français", []),
    ("The water is cold", "L'eau est froide", []), ("She has a car", "Elle a une voiture", []),
    ("The red flower", "La fleur rouge", []), ("Good morning", "Bonjour", []), ("I am hungry", "J'ai faim", []),
    ("The book is interesting", "Le livre est intéressant", []), ("My father works", "Mon père travaille", []),
    ("The sea is blue", "La mer est bleue", []), ("The old bridge", "Le vieux pont", []),
    ("I drink tea", "Je bois du thé", []), ("The children are playing", "Les enfants jouent", []),
    ("It is cold", "Il fait froid", []), ("The green door", "La porte verte", []),
    ("I love music", "J'aime la musique", ["J'adore la musique"]), ("The bread is fresh", "Le pain est frais", []),
    ("Happy birthday", "Joyeux anniversaire", ["Bon anniversaire"]), ("The new school", "La nouvelle école", []),
    ("I live in Paris", "J'habite à Paris", ["Je vis à Paris"]), ("We eat cheese", "Nous mangeons du fromage", []),
    ("The blue sky", "Le ciel bleu", []), ("I don't know", "Je ne sais pas", []),
    ("The beautiful garden", "Le beau jardin", ["Le joli jardin"]), ("Open the window", "Ouvre la fenêtre", ["Ouvrez la fenêtre"]),
    ("The coffee is hot", "Le café est chaud", []), ("My sister reads", "Ma sœur lit", ["Ma soeur lit"]),
    ("The yellow bird", "L'oiseau jaune", []), ("I am tired", "Je suis fatigué", ["Je suis fatiguée"]),
    ("The long road", "La longue route", []), ("He sings well", "Il chante bien", []),
    ("The white horse", "Le cheval blanc", []), ("I have a brother", "J'ai un frère", []),
    ("The library is closed", "La bibliothèque est fermée", []), ("Good night", "Bonne nuit", []),
    ("The moon is bright", "La lune est brillante", []), ("The cold winter", "L'hiver froid", []),
]
FRENCH_T = ["Translate '{e}' into French.", "How do you say '{e}' in French?", "What is '{e}' in French?",
            "Give the French for '{e}'.", "Render '{e}' in French.", "Put '{e}' into French."]


def translation_questions(rng):
    out = []
    for e, s, acc in SPANISH:
        out.append(text_q("spanish", rng.choice(SPANISH_T).format(e=e), s, acc))
    for e, s, acc in FRENCH:
        out.append(text_q("french", rng.choice(FRENCH_T).format(e=e), s, acc))
    return out


# ---------------------------------------------------------------- python
PYTHON = [  # question, answer, env (variable bindings used to check equivalent behaviour)
    ("Give a Python expression that reverses the list items.", "items[::-1]", {"items": [[1, 2, 3, 4, 5], [], [7]]}),
    ("Give a Python slice expression that returns the first three elements of data.", "data[:3]", {"data": [[1, 2, 3, 4, 5], [9, 8]]}),
    ("Write a Python expression for the last element of scores.", "scores[-1]", {"scores": [[4, 8, 15], [2]]}),
    ("Give a Python slice that returns the last two items of values.", "values[-2:]", {"values": [[1, 2, 3, 4], [5]]}),
    ("Write a Python expression that returns the length of the string name.", "len(name)", {"name": ["Alice", ""]}),
    ("Give a Python expression that converts the string text to uppercase.", "text.upper()", {"text": ["hello World"]}),
    ("Write a Python expression that adds up the numbers in nums.", "sum(nums)", {"nums": [[1, 2, 3], [10, -4]]}),
    ("Give a Python expression that returns the largest value in prices.", "max(prices)", {"prices": [[3, 9, 2], [5]]}),
    ("Write a Python expression that splits the string line on commas.", "line.split(',')", {"line": ["a,b,c", "x"]}),
    ("Give a Python expression that joins the list words with single spaces.", "' '.join(words)", {"words": [["hi", "there"], ["a"]]}),
    ("Write a Python expression that returns the items of row from index 2 up to but not including index 5.", "row[2:5]", {"row": [list(range(8)), [1, 2, 3]]}),
    ("Give a Python expression that removes leading and trailing whitespace from s.", "s.strip()", {"s": ["  hi  ", "x"]}),
    ("Write a Python expression that returns a new list with ages sorted in ascending order.", "sorted(ages)", {"ages": [[30, 12, 45], []]}),
    ("Give a Python expression that counts how many times 'a' appears in the string word.", "word.count('a')", {"word": ["banana", "xyz"]}),
    ("Write a Python list comprehension that squares each number in nums.", "[n ** 2 for n in nums]", {"nums": [[1, 2, 3], []]}),
    ("Give a Python expression that returns the unique elements of tags as a set.", "set(tags)", {"tags": [["a", "b", "a"]]}),
    ("Write a Python expression that checks whether 'x' is in the list letters.", "'x' in letters", {"letters": [["a", "x"], ["b"]]}),
    ("Give a Python expression that builds a dictionary from the lists keys and vals.", "dict(zip(keys, vals))", {"keys": [["a", "b"]], "vals": [[1, 2]]}),
    ("Write a Python expression that rounds price to two decimal places.", "round(price, 2)", {"price": [3.14159, 2.5]}),
    ("Give a Python expression for the remainder when total is divided by 7.", "total % 7", {"total": [50, 7, 3]}),
    ("Write a Python expression that converts the string digits to an integer.", "int(digits)", {"digits": ["42", "007"]}),
    ("Give a Python expression that returns the smallest value in temps.", "min(temps)", {"temps": [[3, -1, 7], [4]]}),
    ("Write a Python expression that creates a list of the numbers 0 through 9.", "list(range(10))", {}),
    ("Give a Python expression that converts the string title to lowercase.", "title.lower()", {"title": ["Hello WORLD"]}),
    ("Write a Python expression that returns everything in path except the first character.", "path[1:]", {"path": ["/usr/bin", "a"]}),
    ("Give a Python expression that returns the first character of code.", "code[0]", {"code": ["XYZ", "q"]}),
    ("Write a Python list comprehension that keeps only the even numbers in nums.", "[n for n in nums if n % 2 == 0]", {"nums": [[1, 2, 3, 4], [5]]}),
    ("Give a Python expression that returns the absolute value of delta.", "abs(delta)", {"delta": [-3, 4, 0]}),
    ("Write a Python expression that replaces spaces with underscores in label.", "label.replace(' ', '_')", {"label": ["a b c", "x"]}),
    ("Give a Python expression for the number of keys in the dictionary config.", "len(config)", {"config": [{"a": 1, "b": 2}, {}]}),
    ("Write a Python slice that returns the list cells without its last element.", "cells[:-1]", {"cells": [[1, 2, 3], [1]]}),
    ("Give a Python expression that integer-divides total by 4.", "total // 4", {"total": [17, 8, 3]}),
    ("Write a Python expression that raises base to the power of 3.", "base ** 3", {"base": [2, 5, -1]}),
    ("Give a Python expression that checks whether the string fname ends with '.csv'.", "fname.endswith('.csv')", {"fname": ["a.csv", "b.txt"]}),
    ("Write a Python expression that returns the index of the value 5 in the list nums.", "nums.index(5)", {"nums": [[1, 5, 3], [5]]}),
    ("Give a Python expression that returns a copy of the string msg with its characters reversed.", "msg[::-1]", {"msg": ["hello", ""]}),
]


def python_questions(rng):
    out = []
    for q, a, env in PYTHON:
        out.append(dict(task="python", question=q, answer=a,
                        check=dict(type="python", expected=a, env=env), details=None))
    return out


# ---------------------------------------------------------------- json
JSON_STR = [("name", ["Ada", "Omar", "Lina", "Kai"]), ("color", ["red", "teal", "amber"]), ("city", ["Doha", "Oslo", "Lima"]),
            ("status", ["active", "pending", "closed"]), ("mode", ["dark", "auto", "manual"]), ("unit", ["kg", "cm", "ms"]),
            ("role", ["admin", "guest", "editor"]), ("label", ["urgent", "draft", "final"]), ("animal", ["owl", "fox", "yak"])]
JSON_INT = [("age", [21, 34, 58]), ("score", [88, 12, 95]), ("id", [101, 7, 4402]), ("size", [3, 12, 40]),
            ("year", [1999, 2021, 2010]), ("quantity", [5, 18, 250]), ("rank", [1, 9, 23]), ("port", [8080, 22, 443])]
JSON_BOOL = ["active", "verified", "enabled", "visible", "archived"]
JSON_T2 = ["Write a JSON object with string {k1} {v1} and integer {k2} {v2}.",
           "Produce a JSON object where {k1} is the string \"{v1}\" and {k2} is the integer {v2}.",
           "Give me JSON with a string field {k1} set to {v1} and an integer field {k2} set to {v2}.",
           "Create a JSON object containing {k1} (string, {v1}) and {k2} (integer, {v2})."]
JSON_TB = ["Write a JSON object with boolean {k1} set to {v1} and string {k2} {v2}.",
           "Create JSON where {k1} is the boolean {v1} and {k2} is the string \"{v2}\"."]
JSON_TL = ["Write a JSON object with a key {k} whose value is a list of the integers {vals}.",
           "Produce JSON with {k} as an array of the integers {vals}."]


def json_questions(rng):
    out = []
    seen = set()
    while len(out) < 36:
        kind = rng.choice(["si", "si", "bs", "li"])
        if kind == "si":
            (k1, vs), (k2, vi) = rng.choice(JSON_STR), rng.choice(JSON_INT)
            v1, v2 = rng.choice(vs), rng.choice(vi)
            obj = {k1: v1, k2: v2}
            q = rng.choice(JSON_T2).format(k1=k1, v1=v1, k2=k2, v2=v2)
            spec = {k1: ["str", v1], k2: ["int", v2]}
        elif kind == "bs":
            k1 = rng.choice(JSON_BOOL); b = rng.choice([True, False])
            k2, vs = rng.choice(JSON_STR); v2 = rng.choice(vs)
            obj = {k1: b, k2: v2}
            q = rng.choice(JSON_TB).format(k1=k1, v1=str(b).lower(), k2=k2, v2=v2)
            spec = {k1: ["bool", b], k2: ["str", v2]}
        else:
            k = rng.choice(["values", "ids", "scores", "points", "sizes"])
            vals = rng.sample(range(1, 30), 3)
            obj = {k: vals}
            q = rng.choice(JSON_TL).format(k=k, vals=", ".join(map(str, vals[:-1])) + f", and {vals[-1]}")
            spec = {k: ["list", vals]}
        if q in seen:
            continue
        seen.add(q)
        out.append(dict(task="json", question=q, answer=json.dumps(obj), check=dict(type="json", fields=spec), details=None))
    return out


# ---------------------------------------------------------------- extraction
EXTRACTION = [
    ("gate", "Flight 212 boards at gate C4.", "C4", []), ("room number", "The seminar is in room 305.", "305", []),
    ("date", "The invoice was issued on 12 March 2024.", "12 March 2024", ["March 12, 2024"]),
    ("email address", "Contact Maria at maria.lopez@example.com for details.", "maria.lopez@example.com", []),
    ("price", "The lamp costs $49.99 at the outlet.", "$49.99", ["49.99"]), ("city", "Our next meetup will be held in Lyon.", "Lyon", []),
    ("time", "The webinar starts at 3:30 PM.", "3:30 PM", []), ("order number", "Your order #58213 has shipped.", "58213", ["#58213"]),
    ("speaker's name", "Dr. Priya Raman will give the keynote.", "Priya Raman", ["Dr. Priya Raman"]),
    ("flight number", "Passengers on flight BA287 should proceed to boarding.", "BA287", []),
    ("platform", "The 8:15 train leaves from platform 6.", "6", ["platform 6"]),
    ("color", "She bought a teal backpack for school.", "Teal", []), ("year", "The bridge was completed in 1932.", "1932", []),
    ("phone number", "Call the front desk at 555-0142.", "555-0142", []), ("product code", "Item ZX-44 is out of stock.", "ZX-44", []),
    ("temperature", "The forecast high is 31°C tomorrow.", "31°C", ["31 °C", "31 degrees"]),
    ("locker number", "Your bag is in locker 17.", "17", []), ("percentage", "Sales rose by 12% last quarter.", "12%", ["12 percent"]),
    ("address", "The office moved to 48 Elm Street.", "48 Elm Street", []), ("department", "Please send the form to the Finance department.", "Finance", []),
    ("company", "Lena works as an engineer at Brightwave.", "Brightwave", []), ("account ID", "Account AC-9031 was updated today.", "AC-9031", []),
    ("deadline", "Applications close on June 30.", "June 30", ["30 June"]), ("booking reference", "Your booking reference is QX7P2.", "QX7P2", []),
    ("author", "The novel was written by Chinua Achebe.", "Chinua Achebe", []), ("version", "Please upgrade to version 4.2.1.", "4.2.1", []),
    ("duration", "The hike takes about three hours.", "Three hours", ["3 hours", "about three hours"]),
    ("username", "Log in with the username river_88 to continue.", "river_88", []),
    ("zip code", "Ship it to Springfield, ZIP code 62704.", "62704", []), ("table number", "Your reservation is at table 14.", "14", []),
]
EXTRACT_T = ["Extract the {f} from '{s}'", "What is the {f} in '{s}'", "From '{s}', extract the {f}.", "Pull the {f} out of this sentence: '{s}'",
             "Find the {f} in the text '{s}'", "Identify the {f} mentioned in '{s}'"]


def extraction_questions(rng):
    out = []
    for f, s, a, acc in EXTRACTION:
        t = rng.choice(EXTRACT_T)
        q = t.format(f=f, s=s)
        if q.startswith("What is"):
            q += "?"
        elif not q.endswith("."):
            q += ""  # sentence already ends inside the quote
        out.append(text_q("extraction", q, a, acc))
    return out


# ---------------------------------------------------------------- calendar
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
DAYS_IN = {"January": 31, "March": 31, "April": 30, "May": 31, "June": 30, "July": 31, "August": 31, "September": 30,
           "October": 31, "November": 30, "December": 31}
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
       10: "tenth", 11: "eleventh", 12: "twelfth"}


def calendar_questions(rng):
    out, seen = [], set()
    while len(out) < 44:
        kind = rng.choice(["after", "before", "dow", "dow", "days_in", "nth"])
        if kind == "after":
            m, n = rng.randrange(12), rng.randint(2, 9)
            if (m, n) == (9, 4):
                continue  # evaluation question: four months after October
            q = rng.choice(["Which month comes {n} months after {m}?", "Name the month that is {n} months after {m}.",
                            "If it is {m} now, what month will it be in {n} months?"]).format(n=NUMW[n], m=MONTHS[m])
            a = MONTHS[(m + n) % 12]
        elif kind == "before":
            m, n = rng.randrange(12), rng.randint(2, 9)
            if (m, n) == (1, 4):
                continue
            q = rng.choice(["Which month is {n} months before {m}?", "What month was it {n} months before {m}?"]).format(n=NUMW[n], m=MONTHS[m])
            a = MONTHS[(m - n) % 12]
        elif kind == "dow":
            d, n = rng.randrange(7), rng.randint(2, 11)
            q = rng.choice(["If today is {d}, what day of the week will it be in {n} days?",
                            "What day of the week is {n} days after {d}?",
                            "Starting from {d}, which weekday comes {n} days later?"]).format(d=DAYS[d], n=NUMW[n])
            a = DAYS[(d + n) % 7]
        elif kind == "days_in":
            m = rng.choice(list(DAYS_IN))
            q = rng.choice(["How many days are in {m}?", "How many days does {m} have?"]).format(m=m)
            a = str(DAYS_IN[m])
            if q in seen:
                continue
            seen.add(q)
            out.append(num_q("calendar", q, f"{a} days", a))
            continue
        else:
            k = rng.randint(1, 12)
            q = rng.choice(["What is the {o} month of the year?", "Which month is the {o} month of the year?"]).format(o=ORD[k])
            a = MONTHS[k - 1]
        if q in seen:
            continue
        seen.add(q)
        out.append(text_q("calendar", q, a))
    return out


# ---------------------------------------------------------------- sentiment
SENTIMENT = {
    "Positive": ["I absolutely loved this movie", "The service was fantastic", "Best purchase I've made all year",
                 "The staff were friendly and helpful", "This phone works perfectly", "What a wonderful surprise",
                 "The food was delicious", "I'm thrilled with the results", "Highly recommend this book",
                 "The concert exceeded my expectations", "Great value and fast shipping", "My new chair is so comfortable",
                 "The hotel room was spotless", "This app makes my life easier", "I would happily buy this again"],
    "Negative": ["The package arrived broken", "Terrible customer service", "I regret buying this blender",
                 "The movie was painfully boring", "My order never showed up", "The room smelled awful",
                 "This charger stopped working after a day", "Worst meal I've ever had", "The instructions were confusing and useless",
                 "I'm very disappointed with the quality", "The battery dies way too fast", "Rude staff and long waits",
                 "The shoes fell apart within a week", "Never ordering from here again", "The update made everything slower"],
    "Neutral": ["The meeting is scheduled for Tuesday", "The store opens at 9 a.m.", "The package weighs two kilograms",
                "This book has 300 pages", "The bus stops at the corner", "The product comes in blue and gray",
                "I received the email this morning", "The report is due on Friday", "The museum is on Main Street",
                "The box contains three items", "The train departs at noon", "The jacket is made of cotton",
                "The class lasts one hour", "The restaurant is next to the bank", "The phone has a 6-inch screen"],
}
SENT_T = ["Classify '{t}' as positive, neutral, or negative.", "Is the sentiment of '{t}' positive, negative, or neutral?",
          "Label the sentiment of this review: '{t}' (positive, neutral, or negative).",
          "What is the sentiment of '{t}': positive, negative, or neutral?",
          "Give the sentiment (positive/neutral/negative) of '{t}'."]


def sentiment_questions(rng):
    out = []
    for label, texts in SENTIMENT.items():
        for t in texts:
            out.append(dict(task="sentiment", question=rng.choice(SENT_T).format(t=t), answer=label,
                            check=dict(type="label", expected=label, labels=["positive", "negative", "neutral"]), details=None))
    return out


# ---------------------------------------------------------------- word problems
def word_problem_questions(rng):
    out, seen = [], set()
    names = ["Maya", "Omar", "Lena", "Ravi", "Sara", "Tom", "Aisha", "Jon", "Mei", "Yusuf"]
    while len(out) < 48:
        k = rng.randrange(11)
        n = rng.choice(names)
        if k == 0:
            a, b = rng.randint(20, 90), rng.randint(3, 19)
            q, ans, v = f"{n} has {a} stickers and gives away {b}. How many stickers are left?", f"{a - b} stickers", a - b
        elif k == 1:
            a, b = rng.randint(12, 45), rng.randint(3, 9)
            q, ans, v = f"A bus carries {a} passengers on each trip. How many passengers does it carry in {b} trips?", f"{a * b} passengers", a * b
        elif k == 2:
            a, b = rng.randint(3, 8), rng.randint(6, 12)
            q, ans, v = f"A baker makes {a} trays with {b} muffins on each tray. How many muffins is that?", f"{a * b} muffins", a * b
        elif k == 3:
            a, b = rng.randint(20, 80), rng.randint(3, 15)
            q, ans, v = f"A rope is {a} meters long. If {b} meters are cut off, how many meters remain?", f"{a - b} meters", a - b
        elif k == 4:
            a, b = rng.choice([40, 50, 60, 70, 80, 90]), rng.randint(2, 5)
            q, ans, v = f"A car travels at {a} km/h for {b} hours. How far does it travel?", f"{a * b} km", a * b
        elif k == 5:
            a, b = rng.randint(2, 9), rng.randint(3, 12)
            q, ans, v = f"Pens cost ${a} each. How much do {b} pens cost?", f"${a * b}", a * b
        elif k == 6:
            a, b = rng.randint(4, 12), rng.randint(5, 15)
            q, ans, v = f"A garden has {a} rows with {b} plants in each row. How many plants are there in total?", f"{a * b} plants", a * b
        elif k == 7:
            a, b = rng.randint(15, 60), rng.randint(15, 60)
            q, ans, v = f"{n} read {a} pages on Monday and {b} pages on Tuesday. How many pages were read in total?", f"{a + b} pages", a + b
        elif k == 8:
            b = rng.randint(3, 8); each = rng.randint(3, 9); a = b * each
            q, ans, v = f"{a} cookies are shared equally among {b} children. How many cookies does each child get?", f"{each} cookies", each
        elif k == 9:
            a, b, c = rng.randint(3, 6), rng.randint(6, 12), rng.randint(2, 10)
            q, ans, v = (f"A school buys {a} packs of markers with {b} markers in each pack and then gives away {c} markers. "
                         f"How many markers does it have left?"), f"{a * b - c} markers", a * b - c
        else:
            a, b = rng.randint(10, 40), rng.randint(10, 40)
            q, ans, v = f"There are {a} red marbles and {b} blue marbles in a bag. How many marbles are there altogether?", f"{a + b} marbles", a + b
        if q in seen or v == 108:
            continue
        seen.add(q)
        out.append(num_q("word_problem", q, ans, v))
    return out


# ---------------------------------------------------------------- counting
COUNT_WORDS = ["mountain", "keyboard", "sunflower", "elephant", "notebook", "butterfly", "computer", "pineapple", "rainbow",
               "waterfall", "chocolate", "umbrella", "triangle", "dinosaur", "crocodile", "volcano", "library", "harmony",
               "blanket", "lantern", "orchestra", "telescope", "adventure", "window", "pencil", "garden", "planet",
               "question", "kangaroo", "avocado", "hamburger", "calendar", "strawberry", "tomorrow", "breakfast"]
SENTENCES = ["The quick brown fox jumps over the lazy dog", "We will meet at the old library tomorrow",
             "Rain fell softly on the quiet town", "She sold three paintings at the fair",
             "Please bring your notebook to class", "Birds sing early in the morning",
             "The museum closes at six on Sundays", "My brother plays the violin every evening"]


def count_questions(rng):
    out, seen = [], set()
    letter_t = ["Count the letters in the word '{w}'.", "How many letters does '{w}' have?", "What is the number of letters in '{w}'?",
                "How many letters make up the word {w}?"]
    vowel_t = ["How many vowels are in '{w}'?", "Count the vowels in the word '{w}'."]
    occ_t = ["How many times does the letter '{c}' appear in '{w}'?", "Count how many '{c}'s are in the word '{w}'."]
    word_t = ["How many words are in the sentence '{s}'?", "Count the words in '{s}'."]
    for w in COUNT_WORDS:
        k = rng.random()
        if k < 0.55:
            q, v, unit = rng.choice(letter_t).format(w=w), len(w), "letters"
        elif k < 0.75:
            q, v, unit = rng.choice(vowel_t).format(w=w), sum(ch in "aeiou" for ch in w), "vowels"
        else:
            c = rng.choice(sorted(set(w)))
            q, v, unit = rng.choice(occ_t).format(c=c, w=w), w.count(c), "times"
        out.append(num_q("count", q, str(v), v))
    for s in SENTENCES:
        q = rng.choice(word_t).format(s=s)
        out.append(num_q("count", q, str(len(s.split())), len(s.split())))
    return out


# ---------------------------------------------------------------- sorting
SORT_POOLS = [["apple", "mango", "banana", "kiwi", "grape", "cherry", "peach", "lemon", "plum"],
              ["zebra", "otter", "falcon", "lynx", "badger", "heron", "camel", "moose"],
              ["violet", "crimson", "amber", "indigo", "olive", "scarlet", "beige"],
              ["Tokyo", "Lagos", "Denver", "Quito", "Madrid", "Hanoi", "Cairo"],
              ["oak", "maple", "willow", "spruce", "elm", "ash", "poplar"],
              ["piano", "flute", "guitar", "drum", "harp", "violin", "cello"],
              ["copper", "silver", "nickel", "zinc", "iron", "tin"]]


def sorting_questions(rng):
    out, seen = [], set()
    alpha_t = ["Put {x} in alphabetical order.", "Sort {x} alphabetically.", "Arrange {x} in alphabetical order.",
               "List {x} in A-to-Z order."]
    rev_t = ["Put {x} in reverse alphabetical order.", "Sort {x} from Z to A."]
    num_t = ["Sort these numbers from smallest to largest: {x}.", "Arrange {x} in ascending order.",
             "Order {x} from largest to smallest."]
    while len(out) < 40:
        k = rng.random()
        if k < 0.6:
            words = rng.sample(rng.choice(SORT_POOLS), rng.choice([3, 3, 4]))
            if set(w.lower() for w in words) & {"pine", "birch", "cedar"}:
                continue
            x = ", ".join(words[:-1]) + (", and " if len(words) > 2 else " and ") + words[-1]
            q = rng.choice(alpha_t).format(x=x)
            order = sorted(words, key=str.lower)
        elif k < 0.75:
            words = rng.sample(rng.choice(SORT_POOLS), 3)
            x = ", ".join(words[:-1]) + ", and " + words[-1]
            q = rng.choice(rev_t).format(x=x)
            order = sorted(words, key=str.lower, reverse=True)
        else:
            nums = rng.sample(range(2, 99), rng.choice([4, 5]))
            x = ", ".join(map(str, nums))
            t = rng.choice(num_t)
            q = t.format(x=x)
            order = [str(n) for n in sorted(nums, reverse="largest to smallest" in t)]
        if q in seen:
            continue
        seen.add(q)
        out.append(dict(task="sorting", question=q, answer=", ".join(order),
                        check=dict(type="order", expected=", ".join(order), items=order), details=None))
    return out


# ---------------------------------------------------------------- general knowledge
GENERAL = [
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare", ["Shakespeare"]),
    ("How many continents are there?", "Seven", ["7"]),
    ("What is the longest river in Africa?", "The Nile", ["Nile"]),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["da Vinci", "Leonardo"]),
    ("How many sides does a hexagon have?", "Six", ["6"]),
    ("Which ocean is the largest?", "The Pacific Ocean", ["Pacific"]),
    ("What language is mainly spoken in Brazil?", "Portuguese", []),
    ("What is the currency of Japan?", "The yen", ["yen"]),
    ("Who developed the theory of relativity?", "Albert Einstein", ["Einstein"]),
    ("What is the tallest mountain in the world?", "Mount Everest", ["Everest"]),
    ("What do bees make?", "Honey", []),
    ("Which organ pumps blood through the body?", "The heart", ["heart"]),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", ["Armstrong"]),
    ("What is the largest mammal?", "The blue whale", ["blue whale"]),
    ("In which country are the pyramids of Giza?", "Egypt", []),
    ("How many legs does a spider have?", "Eight", ["8"]),
    ("Which instrument has 88 keys?", "The piano", ["piano"]),
    ("What is the smallest prime number?", "2", ["two"]),
    ("Who wrote the novel '1984'?", "George Orwell", ["Orwell"]),
    ("Which continent is Kenya in?", "Africa", []),
    ("What is the fastest land animal?", "The cheetah", ["cheetah"]),
    ("Which metal is liquid at room temperature?", "Mercury", []),
    ("What is the name of our galaxy?", "The Milky Way", ["Milky Way"]),
    ("What color do you get by mixing blue and yellow?", "Green", []),
    ("What color do you get when you mix red and white?", "Pink", []),
    ("What is the past tense of 'run'?", "Ran", []),
    ("What is the past tense of 'go'?", "Went", []),
    ("What is the comparative form of 'good'?", "Better", []),
    ("What is the superlative form of 'bad'?", "Worst", []),
    ("What does CPU stand for?", "Central processing unit", []),
    ("What does HTML stand for?", "HyperText Markup Language", ["Hypertext Markup Language"]),
    ("What does DNA stand for?", "Deoxyribonucleic acid", []),
    ("How many bits are in a byte?", "8", ["eight"]),
    ("What is the binary representation of the number 5?", "101", []),
    ("What is the next prime number after 7?", "11", ["eleven"]),
    ("How many degrees are in a right angle?", "90 degrees", ["90"]),
    ("How many degrees are in a full circle?", "360 degrees", ["360"]),
    ("What is the value of pi rounded to two decimal places?", "3.14", []),
    ("In what year did World War II end?", "1945", []),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen", ["Austen"]),
    ("Which planet is famous for its large rings?", "Saturn", []),
    ("What is the largest country by area?", "Russia", []),
    ("What is the smallest country in the world by area?", "Vatican City", ["Vatican"]),
    ("Which gas do humans breathe out?", "Carbon dioxide", ["CO2"]),
    ("How many players does a soccer team have on the field?", "11", ["eleven"]),
    ("What is the opposite of 'ancient'?", "Modern", []),
    ("What is the opposite of 'maximum'?", "Minimum", []),
    ("What is the opposite of 'visible'?", "Invisible", []),
    ("Which shape has three sides?", "A triangle", ["triangle"]),
    ("What is the main ingredient in guacamole?", "Avocado", []),
    ("How many minutes are in an hour?", "60", ["sixty"]),
    ("What is the largest hot desert in the world?", "The Sahara", ["Sahara"]),
    ("Which country gifted the Statue of Liberty to the United States?", "France", []),
    ("What is the square root of 144?", "12", ["twelve"]),
    ("How many sides does an octagon have?", "Eight", ["8"]),
    ("What is the chemical symbol for table salt's metal, sodium?", "Na", []),
    ("Who is known for the laws of motion and universal gravitation?", "Isaac Newton", ["Newton"]),
    ("Which language has the most native speakers worldwide?", "Mandarin Chinese", ["Mandarin", "Chinese"]),
    ("What is the freezing point of water in degrees Fahrenheit?", "32°F", ["32 °F", "32 degrees"]),
    ("What is 25% of 80?", "20", ["twenty"]),
    ("What is the capital letter form of 'q'?", "Q", []),
    ("Which programming language is named after a comedy group?", "Python", []),
    ("What does 'www' stand for in a web address?", "World Wide Web", []),
    ("How many hours are in two days?", "48 hours", ["48"]),
]


def general_questions(rng):
    out = []
    for q, a, acc in GENERAL:
        if any(ch.isdigit() for ch in a) and len(a.split()) <= 2:
            num = "".join(ch for ch in a.split()[0] if ch.isdigit() or ch == ".")
            out.append(num_q("general", q, a, num, acc))
        else:
            out.append(text_q("general", q, a, acc))
    return out


def all_questions(seed):
    rng = random.Random(seed)
    groups = [units_questions, arithmetic_questions, capital_questions, element_questions, science_questions,
              plural_questions, translation_questions, python_questions, json_questions, extraction_questions,
              calendar_questions, sentiment_questions, word_problem_questions, count_questions, sorting_questions,
              general_questions]
    qs = []
    for g in groups:
        qs.extend(g(rng))
    return qs
