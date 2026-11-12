"""TravelPlanner (OSU-NLP-Group/TravelPlanner, HF `osunlp/TravelPlanner`), sole-planning mode.

The prompt is TravelPlanner's own sole-planning PLANNER_INSTRUCTION (agents/prompts.py) filled with the query and the
dataset's `reference_information`. The official pipeline converts the free-text plan to JSON with GPT-4
(postprocess/openai_request.py::build_plan_format_conversion_prompt); we can't call an API, so the prompt's example is
rewritten in that official JSON format and the model is asked to output the JSON list directly in a ```json block
(the formatting rules are copied from the official parsing prompt).

Grading: the official commonsense + hard constraint checks (evaluation/commonsense_constraint.py,
evaluation/hard_constraint.py, evaluation/eval.py), copied below with the database lookups pointed at the official
database (Google Drive zip, ~59MB, ~295MB unzipped). "correct" = their per-query final pass (all commonsense checks
pass AND hard checks were run and none failed).
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import zipfile

DEPS = ["datasets", "pandas"]

DATA_DIR = ("/private/tmp/claude-501/-Users-shriyanreyya-Documents-SPAR-reward-seeking-rl/"
            "0cb16837-dc48-45f7-beaa-e9248752d418/scratchpad/nov11_data/travelplanner")
DB_DIR = os.path.join(DATA_DIR, "dbx", "database")
DB_URL = ("https://drive.usercontent.google.com/download?id=1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE"
          "&export=download&confirm=t")  # official DB link from the TravelPlanner README

TASK = "travelplanner_sole"


def tasks() -> dict[str, dict]:
    return {TASK: {
        "skill": "long-context constrained planning (travel itinerary under budget/commonsense/hard constraints)",
        "source": "TravelPlanner (osunlp/TravelPlanner) validation split, sole-planning mode",
        "notes": "Official planner prompt + reference info; model emits the official JSON plan directly (no GPT "
                 "parsing step). correct = official final pass (all commonsense + hard constraints). Extras: "
                 "commonsense/hard micro pass fractions and macro passes.",
    }}


# --------------------------------------------------------------------------------------------------------------------
# Prompt: agents/prompts.py PLANNER_INSTRUCTION, with the example rendered in the official JSON format (the format
# of postprocess/openai_request.py's parsing prompt) and an explicit output-format instruction.
# --------------------------------------------------------------------------------------------------------------------
_EXAMPLE_PLAN = [
    {"days": 1, "current_city": "from Ithaca to Charlotte",
     "transportation": "Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46",
     "breakfast": "Nagaland's Kitchen, Charlotte", "attraction": "The Charlotte Museum of History, Charlotte;",
     "lunch": "Cafe Maple Street, Charlotte", "dinner": "Bombay Vada Pav, Charlotte",
     "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"},
    {"days": 2, "current_city": "Charlotte", "transportation": "-", "breakfast": "Olive Tree Cafe, Charlotte",
     "attraction": "The Mint Museum, Charlotte;Romare Bearden Park, Charlotte;",
     "lunch": "Birbal Ji Dhaba, Charlotte", "dinner": "Pind Balluchi, Charlotte",
     "accommodation": "Affordable Spacious Refurbished Room in Bushwick!, Charlotte"},
    {"days": 3, "current_city": "from Charlotte to Ithaca",
     "transportation": "Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26",
     "breakfast": "Subway, Charlotte", "attraction": "Books Monument, Charlotte;", "lunch": "Olive Tree Cafe, Charlotte",
     "dinner": "Kylin Skybar, Charlotte", "accommodation": "-"},
]

PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information and query, please give me a detailed plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plan should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with commonsense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the 'current_city' field as in the example (i.e., from A to B).

Output format: give the final plan as a JSON list inside a ```json code block, with one object per day and exactly the keys "days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation". Format every restaurant, attraction and accommodation as 'Name, City'. Separate attractions with ';' and end each attraction with ';' (e.g. 'A, City;B, City;'). Write flights as 'Flight Number: F0123456, from A to B, Departure Time: hh:mm, Arrival Time: hh:mm', and ground transport as 'Self-driving, from A to B' or 'Taxi, from A to B'. Use '-' for information that is unnecessary. Do not include '$' symbols.

***** Example *****
Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Travel Plan:
```json
{example}
```

***** Example Ends *****

Given information: {text}
Query: {query}
Travel Plan:"""


def _build_prompt(text: str, query: str) -> str:
    return (PLANNER_INSTRUCTION.replace("{example}", json.dumps(_EXAMPLE_PLAN, indent=4, ensure_ascii=False))
            .replace("{text}", text).replace("{query}", query))


# --------------------------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------------------------
_QUERY_KEYS = ["org", "dest", "days", "visiting_city_number", "date", "people_number", "local_constraint", "budget",
               "query", "level"]


def _query_data(row: dict) -> dict:
    q = {k: row[k] for k in _QUERY_KEYS}
    if isinstance(q["local_constraint"], str):
        q["local_constraint"] = ast.literal_eval(q["local_constraint"])
    if isinstance(q["date"], str):
        q["date"] = ast.literal_eval(q["date"])
    return q


def _ensure_db() -> None:
    if os.path.exists(os.path.join(DB_DIR, "flights", "clean_Flights_2022.csv")):
        return
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True)
    zpath = os.path.join(DATA_DIR, "db.zip")
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(DB_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():  # extract only the expected data files (no path traversal)
            if name.startswith("database/") and not name.endswith("/") and "__MACOSX" not in name and ".." not in name:
                z.extract(name, os.path.join(DATA_DIR, "dbx"))


def load(task: str, n: int = 100, seed: int = 0) -> list[dict]:
    assert task == TASK, task
    cache = os.path.join(DATA_DIR, f"problems_{task}_n{n}_s{seed}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    from datasets import load_dataset
    rows = list(load_dataset("osunlp/TravelPlanner", "validation")["validation"])
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    probs = []
    for i in idx:
        r = rows[i]
        probs.append({"id": f"tp_val_{i}", "prompt": _build_prompt(r["reference_information"], r["query"]),
                      "query_data": _query_data(r)})
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(probs, f)
    return probs


# --------------------------------------------------------------------------------------------------------------------
# Database (tools/*/apis.py): same columns and dropna as the official loaders.
# --------------------------------------------------------------------------------------------------------------------
_DB: dict = {}


def _db():
    if not _DB:
        import pandas as pd
        _ensure_db()
        _DB["flight"] = pd.read_csv(os.path.join(DB_DIR, "flights/clean_Flights_2022.csv")).dropna()[
            ['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime', 'FlightDate', 'OriginCityName',
             'DestCityName', 'Distance']]
        _DB["accommodation"] = pd.read_csv(os.path.join(DB_DIR, "accommodations/clean_accommodations_2022.csv")).dropna()[
            ['NAME', 'price', 'room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number',
             'city']]
        _DB["restaurants"] = pd.read_csv(os.path.join(DB_DIR, "restaurants/clean_restaurant_2022.csv")).dropna()[
            ['Name', 'Average Cost', 'Cuisines', 'Aggregate Rating', 'City']]
        _DB["distance"] = pd.read_csv(os.path.join(DB_DIR, "googleDistanceMatrix/distance.csv"))
        _DB["attractions"] = pd.read_csv(os.path.join(DB_DIR, "attractions/attractions.csv")).dropna()[
            ['Name', 'Latitude', 'Longitude', 'Address', 'Phone', 'Website', "City"]]
        lines = open(os.path.join(DB_DIR, "background/citySet_with_states.txt")).read().split('\n')
        _DB["city_state_map"] = {x: y for x, y in [u.split('\t') for u in lines if u.strip()]}
    return _DB


# utils/func.py
def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s


def get_valid_name_city(info):
    pattern = r'(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$'
    match = re.search(pattern, info)
    if match:
        return match.group(1).strip(), extract_before_parenthesis(match.group(2).strip()).strip()
    return "-", "-"


# tools/googleDistanceMatrix/apis.py::run_for_evaluation
def _distance_eval(origin, destination, mode='driving'):
    import numpy as np
    data = _db()["distance"]
    origin = extract_before_parenthesis(origin)
    destination = extract_before_parenthesis(destination)
    info = {"origin": origin, "destination": destination, "cost": None, "duration": None, "distance": None}
    response = data[(data['origin'] == origin) & (data['destination'] == destination)]
    if len(response) > 0:
        if response['duration'].values[0] is None or response['distance'].values[0] is None or \
                response['duration'].values[0] is np.nan or response['distance'].values[0] is np.nan:
            return info
        info["duration"] = response['duration'].values[0]
        info["distance"] = response['distance'].values[0]
        if 'day' not in info["duration"]:
            # official code uses eval(); float() on the same cleaned string (distance strings are numeric)
            km = float(info["distance"].replace("km", "").replace(",", ""))
            if 'driving' in mode:
                info["cost"] = int(km * 0.05)
            elif mode == "taxi":
                info["cost"] = int(km)
    return info


def _rest_lookup(name, city):
    r = _db()["restaurants"]
    return r[(r['Name'].astype(str).str.contains(re.escape(name))) & (r['City'] == city)]


def _acc_lookup(name, city):
    a = _db()["accommodation"]
    return a[(a['NAME'].astype(str).str.contains(re.escape(name))) & (a['city'] == city)]


# --------------------------------------------------------------------------------------------------------------------
# evaluation/commonsense_constraint.py (verbatim logic)
# --------------------------------------------------------------------------------------------------------------------
def count_consecutive_values(lst):
    if not lst:
        return []
    result = []
    current_string = lst[0]
    count = 1
    for i in range(1, len(lst)):
        if lst[i] == current_string:
            count += 1
        else:
            result.append((current_string, count))
            current_string = lst[i]
            count = 1
    result.append((current_string, count))
    return result


def transportation_match(text: str):
    if 'taxi' in text.lower():
        return 'Taxi'
    elif 'self-driving' in text.lower():
        return 'Self-driving'
    elif 'flight' in text.lower():
        return 'Flight'


def extract_from_to(text: str):
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)


def is_valid_city_sequence(city_list):
    if len(city_list) < 3:
        return False
    visited_cities = set()
    i = 0
    while i < len(city_list):
        city = city_list[i]
        if city in visited_cities and (i != 0 and i != len(city_list) - 1):
            return False
        count = 0
        while i < len(city_list) and city_list[i] == city:
            count += 1
            i += 1
        if count == 1 and 0 < i - 1 < len(city_list) - 1:
            return False
        visited_cities.add(city)
    return True


def is_reasonable_visiting_city(question, tested_data):
    city_state_map = _db()["city_state_map"]
    city_list = []
    for i in range(min(question['days'], len(tested_data))):
        city_value = tested_data[i]['current_city']
        if 'from' in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i == 0 and city1 != question['org']:
                return False, f"The first day's city should be {question['org']}."
            city_list += [city1, city2]
        else:
            city_list.append(extract_before_parenthesis(city_value))
    if city_list[0] != city_list[-1]:
        return False, "The trip should be a closed circle."
    if not is_valid_city_sequence(city_list):
        return False, "The city sequence is invalid."
    for idx, city in enumerate(city_list):
        if city not in city_state_map:
            return False, f"{city} is not a valid city."
        if idx not in [0, len(city_list) - 1] and question['days'] > 3 and city_state_map[city] != question['dest']:
            return False, f"{city} is not in {question['dest']}."
    return True, None


def is_valid_restaurants(question, tested_data):
    restaurants_list = []
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        for meal in ['breakfast', 'lunch', 'dinner']:
            if meal in unit and unit[meal] and unit[meal] != '-':
                if unit[meal] not in restaurants_list:
                    restaurants_list.append(unit[meal])
                else:
                    return False, f"The restaurant in day {i + 1} {meal} is repeated."
    return True, None


def is_valid_attractions(question, tested_data):
    attractions_list = []
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            for attraction in unit['attraction'].split(';')[:-1]:
                if attraction not in attractions_list:
                    attractions_list.append(attraction)
                else:
                    return False, f"The attraction '{attraction}' in day {i + 1} is repeated."
    return True, None


def is_valid_transportation_cs(question, tested_data):
    if tested_data[0]['transportation'] and tested_data[0]['transportation'] != '-':
        transportation_list = [transportation_match(tested_data[0]['transportation'])]
    else:
        return False, "The transportation in day 1 should not be empty."
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if 'transportation' in unit and unit['transportation'] and unit['transportation'] != '-':
            transportation_list.append(transportation_match(unit['transportation']))
    if (('Self-driving' in transportation_list) and ('Flight' in transportation_list)) or (
            ('Taxi' in transportation_list) and ('Self-driving' in transportation_list)):
        return False, "The transportation is conflicting."
    return True, None


def is_valid_information_in_current_city(question, tested_data):
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        current_city = unit['current_city']
        if 'from' in current_city:
            city1, city2 = extract_from_to(current_city)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            final_city_list = [city1, city2]
        else:
            final_city_list = [extract_before_parenthesis(current_city)]
        if 'transportation' in unit and unit['transportation'] and unit['transportation'] != '-':
            for city in final_city_list:
                if city not in unit['transportation']:
                    return False, f"The transportation in day {i + 1} is invalid city choice."
        for meal in ['breakfast', 'lunch', 'dinner']:
            if meal in unit and unit[meal] and unit[meal] != '-':
                if not any(city in unit[meal] for city in final_city_list):
                    return False, f"The {meal} in day {i + 1} is invalid city choice."
        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            for attraction in unit['attraction'].split(';')[:-1]:
                if not any(city in attraction for city in final_city_list):
                    return False, f"The attraction in day {i + 1} is invalid city choice."
        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            if final_city_list[-1] not in unit['accommodation']:
                return False, f"The accommodation in day {i + 1} is invalid city choice."
    return True, None


def is_valid_information_in_sandbox(question, tested_data):
    db = _db()
    flight, attractions = db["flight"], db["attractions"]
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city is None or dest_city is None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            if 'flight number' in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    raise ValueError("The transportation {} in day {} can not be parsed.".format(value, i + 1))
                if len(flight[(flight['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]) & (
                        flight['OriginCityName'] == org_city) & (flight['DestCityName'] == dest_city)]) < 1:
                    return False, f"The flight number in day {i + 1} is invalid in the sandbox."
            elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                try:
                    org_city = extract_before_parenthesis(org_city)
                    dest_city = extract_before_parenthesis(dest_city)
                except TypeError:
                    org_city = '-'
                    dest_city = '-'
                if 'self-driving' in value.lower():
                    if _distance_eval(org_city, dest_city, mode='self-driving')['cost'] is None:
                        return False, f"The self-driving in day {i + 1} is invalid in the sandbox."
                else:
                    if _distance_eval(org_city, dest_city, mode='taxi')['cost'] is None:
                        return False, f"The taxi in day {i + 1} is invalid in the sandbox."
        for meal in ['breakfast', 'lunch', 'dinner']:
            if meal in unit and unit[meal] and unit[meal] != '-':
                name, city = get_valid_name_city(unit[meal])
                if len(_rest_lookup(name, city)) < 1:
                    return False, f"The {meal} in day {i + 1} is invalid in the sandbox."
        if 'attraction' in unit and unit['attraction'] and unit['attraction'] != '-':
            for attraction in unit['attraction'].split(';')[:-1]:
                name, city = get_valid_name_city(attraction)
                if len(attractions[(attractions['Name'].astype(str).str.contains(re.escape(name))) & (
                        attractions['City'] == city)]) < 1:
                    return False, f"The attraction {attraction} in day {i + 1} is invalid in the sandbox."
        if 'accommodation' in unit and unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            if len(_acc_lookup(name, city)) < 1:
                return False, f"The accommodation in day {i + 1} is invalid in the sandbox."
    return True, None


def is_valid_accommodaton(question, tested_data):
    data = []
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if 'accommodation' not in unit:
            return False, "No Accommodation Info."
        data.append(unit['accommodation'])
    for unit in count_consecutive_values(data):
        if unit and unit[0] not in ['-', '']:
            name, city = get_valid_name_city(unit[0])
            res = _acc_lookup(name, city)
            if len(res) == 1 and unit[1] < res.iloc[0]['minimum nights']:
                return False, f"The accommodation {unit[0]} do not obey the minumum nights rule."
    return True, None


def is_valid_visiting_city_number(question, tested_data):
    city_set = set()
    for i in range(min(question['days'], len(tested_data))):
        city_value = tested_data[i]['current_city']
        if 'from' in city_value:
            city1, city2 = extract_from_to(city_value)
            city1 = extract_before_parenthesis(city1)
            city2 = extract_before_parenthesis(city2)
            if i == 0 and city1 != question['org']:
                return False, f"The first day's city should be {question['org']}."
            city_set.add(city1)
            city_set.add(city2)
        else:
            city_set.add(extract_before_parenthesis(city_value))
    city_set.discard(question['org'])
    if len(city_set) != question['visiting_city_number']:
        return False, f"The number of visiting cities should be {question['visiting_city_number']}."
    return True, None


def is_valid_days(question, tested_data):
    lens = 0
    for i in range(min(question['days'], len(tested_data))):
        if tested_data[i] != {} and tested_data[i][
                'current_city'] != "You don't need to fill in the information for this or later days.":
            lens += 1
    if lens != question['days']:
        return False, f"The number of days should be {question['days']}."
    return True, None


def is_not_absent(question, tested_data):
    needed_info = 6 * question['days']
    total_valid_info = 0
    if not is_valid_days(question, tested_data)[0]:
        return False, "Invalid Days"
    if not is_valid_visiting_city_number(question, tested_data)[0]:
        return False, "Invalid City Number"
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        for k in ['transportation', 'breakfast', 'lunch', 'dinner', 'attraction', 'accommodation']:
            if k not in unit:
                return False, f"No {k} Info."
        if ('from ' in unit['current_city'] or 'to ' in unit['current_city']) and unit['transportation'] in ['', '-']:
            return False, f"No transportation in day {i + 1} is not allowed."
        if ('from ' not in unit['current_city'] and ' to ' not in unit['current_city']) and unit['attraction'] in ['', '-']:
            return False, f"No attaction in day {i + 1} is not allowed."
        if i != question['days'] - 1 and unit['accommodation'] in ['', '-']:
            return False, f"No accommodation in day {i + 1} is not allowed."
        if (unit['breakfast'] in ['', '-'] or unit['lunch'] in ['', '-'] or unit['dinner'] in ['', '-']) and \
                'from ' not in unit['current_city']:
            return False, f"No meal in day {i + 1} is not allowed."
        for key in unit:
            if unit[key] and unit[key] != '-':
                total_valid_info += 1
    if total_valid_info * 1.0 / needed_info < 0.5:
        return False, "The absent information is more than 50%."
    return True, None


def commonsense_eval(query_data, tested_data):
    return {
        'is_reasonable_visiting_city': is_reasonable_visiting_city(query_data, tested_data),
        'is_valid_restaurants': is_valid_restaurants(query_data, tested_data),
        'is_valid_attractions': is_valid_attractions(query_data, tested_data),
        'is_valid_accommodation': is_valid_accommodaton(query_data, tested_data),
        'is_valid_transportation': is_valid_transportation_cs(query_data, tested_data),
        'is_valid_information_in_current_city': is_valid_information_in_current_city(query_data, tested_data),
        'is_valid_information_in_sandbox': is_valid_information_in_sandbox(query_data, tested_data),
        'is_not_absent': is_not_absent(query_data, tested_data),
    }


# --------------------------------------------------------------------------------------------------------------------
# evaluation/hard_constraint.py (verbatim logic)
# --------------------------------------------------------------------------------------------------------------------
def get_total_cost(question, tested_data):
    flight = _db()["flight"]
    total_cost = 0
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            org_city, dest_city = extract_from_to(value)
            if org_city is None or dest_city is None:
                org_city, dest_city = extract_from_to(unit['current_city'])
            if org_city is None or dest_city is None:
                pass
            else:
                if 'flight number' in value.lower():
                    res = flight[flight['Flight Number'] == value.split('Flight Number: ')[1].split(',')[0]]
                    if len(res) > 0:
                        total_cost += res['Price'].values[0] * question['people_number']
                elif 'self-driving' in value.lower() or 'taxi' in value.lower():
                    if 'self-driving' in value.lower():
                        cost = _distance_eval(org_city, dest_city, 'self-driving')['cost']
                        total_cost += cost * math.ceil(question['people_number'] * 1.0 / 5)
                    else:
                        cost = _distance_eval(org_city, dest_city, 'taxi')['cost']
                        total_cost += cost * math.ceil(question['people_number'] * 1.0 / 4)
        for meal in ['breakfast', 'lunch', 'dinner']:
            if unit[meal] and unit[meal] != '-':
                name, city = get_valid_name_city(unit[meal])
                res = _rest_lookup(name, city)
                if len(res) > 0:
                    total_cost += res['Average Cost'].values[0] * question['people_number']
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = _acc_lookup(name, city)
            if len(res) > 0:
                total_cost += res['price'].values[0] * math.ceil(
                    question['people_number'] * 1.0 / res['maximum occupancy'].values[0])
    return total_cost


def is_valid_room_rule(question, tested_data):
    rule = question['local_constraint']['house rule']
    if rule is None:
        return None, None
    neg = {'smoking': 'No smoking', 'parties': 'No parties', 'children under 10': 'No children under 10',
           'visitors': 'No visitors', 'pets': 'No pets'}
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = _acc_lookup(name, city)
            if len(res) > 0 and rule in neg and neg[rule] in str(res['house_rules'].values[0]):
                return False, f"The house rule should be {rule}."
    return True, None


def is_valid_cuisine(question, tested_data):
    cuisine_set = set()
    if question['local_constraint']['cuisine']:
        for i in range(min(question['days'], len(tested_data))):
            unit = tested_data[i]
            # NB: the official code `continue`s the whole day when a meal is in the origin city; kept as-is.
            for meal in ['breakfast', 'lunch', 'dinner']:
                if unit[meal] and unit[meal] != '-':
                    name, city = get_valid_name_city(unit[meal])
                    if city == question['org']:
                        break
                    res = _rest_lookup(name, city)
                    if len(res) > 0:
                        for cuisine in question['local_constraint']['cuisine']:
                            if cuisine in res.iloc[0]['Cuisines']:
                                cuisine_set.add(cuisine)
        if len(cuisine_set) == len(question['local_constraint']['cuisine']):
            return True, None
        for cuisine in question['local_constraint']['cuisine']:
            if cuisine not in cuisine_set:
                return False, f"The cuisine {cuisine} is not satisfied."
    else:
        return None, None


def is_valid_transportation_hard(question, tested_data):
    t = question['local_constraint']['transportation']
    if t is None:
        return None, None
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['transportation'] and unit['transportation'] != '-':
            value = unit['transportation']
            if t == 'no flight' and 'Flight' in value:
                return False, f"The transportation should not be {t}."
            elif t == 'no self-driving' and 'Self-driving' in value:
                return False, f"The transportation should not be {t}."
    return True, None


def is_valid_room_type(question, tested_data):
    rt = question['local_constraint']['room type']
    if rt is None:
        return None, None
    for i in range(min(question['days'], len(tested_data))):
        unit = tested_data[i]
        if unit['accommodation'] and unit['accommodation'] != '-':
            name, city = get_valid_name_city(unit['accommodation'])
            res = _acc_lookup(name, city)
            if len(res) > 0:
                v = res['room type'].values[0]
                if rt == 'not shared room' and v == 'Shared room':
                    return False, f"The room type should be {rt}."
                elif rt == 'shared room' and v != 'Shared room':
                    return False, f"The room type should be {rt}."
                elif rt == 'private room' and v != 'Private room':
                    return False, f"The room type should be {rt}."
                elif rt == 'entire room' and v != 'Entire home/apt':
                    return False, f"The room type should be {rt}."
    return True, None


def hard_eval(query_data, tested_data):
    return {
        'valid_cuisine': is_valid_cuisine(query_data, tested_data),
        'valid_room_rule': is_valid_room_rule(query_data, tested_data),
        'valid_transportation': is_valid_transportation_hard(query_data, tested_data),
        'valid_room_type': is_valid_room_type(query_data, tested_data),
        'valid_cost': (bool(get_total_cost(query_data, tested_data) <= query_data['budget']), None),
    }


# --------------------------------------------------------------------------------------------------------------------
# Parsing + grading
# --------------------------------------------------------------------------------------------------------------------
_PLAN_KEYS = ["days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]


def parse_plan(final: str):
    """Last ```json block (else last bare [...] list) -> list of day dicts in the official format, or None."""
    if not final:
        return None
    blocks = re.findall(r"```json\s*(.*?)```", final, flags=re.S | re.I)
    if not blocks:
        blocks = re.findall(r"```\s*(\[.*?\])\s*```", final, flags=re.S)
    plan = None
    for b in reversed(blocks):
        try:
            plan = json.loads(b.strip())
            break
        except Exception:
            continue
    if isinstance(plan, dict):  # tolerate {"plan": [...]}
        plan = next((v for v in plan.values() if isinstance(v, list)), None)
    if not isinstance(plan, list) or not plan or not all(isinstance(u, dict) for u in plan):
        return None
    out = []
    for u in plan:
        d = {}
        for k, v in u.items():
            if k == "days":
                d[k] = v
            else:
                d[k] = '-' if v is None else (v if isinstance(v, str) else str(v))
        # The official parser output ends every attraction with ';' (the checker drops the last split piece).
        a = d.get("attraction")
        if isinstance(a, str) and a.strip() not in ("", "-") and not a.rstrip().endswith(";"):
            d["attraction"] = a.rstrip() + ";"
        out.append(d)
    return out


def _evaluate(query_data: dict, plan: list) -> dict:
    """Per-query version of evaluation/eval.py::eval_score."""
    try:
        cs = commonsense_eval(query_data, plan)
    except Exception as e:  # malformed plan that crashes the official checker -> fails
        return {"correct": False, "error": f"commonsense: {type(e).__name__}: {e}"}
    hard = None
    err = None
    if cs['is_not_absent'][0] and cs['is_valid_information_in_sandbox'][0]:
        try:
            hard = hard_eval(query_data, plan)
        except Exception as e:
            err = f"hard: {type(e).__name__}: {e}"
    cs_pass = all(v[0] is None or bool(v[0]) for v in cs.values())
    hard_pass = hard is not None and all(v[0] is None or bool(v[0]) for v in hard.values())
    # micro: fraction of the 8 commonsense checks passed; fraction of applicable hard constraints passed
    # (budget + each non-None local constraint; counted as failed when hard checks weren't run, as in eval.py).
    lc = query_data['local_constraint']
    applicable = ['valid_cost'] + [m for k, m in [('house rule', 'valid_room_rule'), ('cuisine', 'valid_cuisine'),
                                                  ('room type', 'valid_room_type'),
                                                  ('transportation', 'valid_transportation')] if lc.get(k)]
    hard_true = sum(1 for k in applicable if hard is not None and hard[k][0] is True)
    return {
        "correct": bool(cs_pass and hard_pass),
        "commonsense_macro_pass": bool(cs_pass),
        "hard_macro_pass": bool(hard_pass),
        "commonsense_micro_pass": sum(1 for v in cs.values() if v[0]) / len(cs),
        "hard_micro_pass": hard_true / len(applicable),
        "commonsense": {k: [bool(v[0]) if v[0] is not None else None, v[1]] for k, v in cs.items()},
        "hard": None if hard is None else {k: [bool(v[0]) if v[0] is not None else None, v[1]] for k, v in hard.items()},
        **({"error": err} if err else {}),
    }


def grade(task: str, problem: dict, final: str) -> dict:
    plan = parse_plan(final)
    if plan is None:
        return {"correct": False, "answered": False, "commonsense_micro_pass": 0.0, "hard_micro_pass": 0.0,
                "commonsense_macro_pass": False, "hard_macro_pass": False}
    res = _evaluate(problem["query_data"], plan)
    res.setdefault("commonsense_micro_pass", 0.0)
    res.setdefault("hard_micro_pass", 0.0)
    res.setdefault("commonsense_macro_pass", False)
    res.setdefault("hard_macro_pass", False)
    return {"answered": True, "n_days_parsed": len(plan), **res}


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    probs = load(TASK, n=100)
    print(f"{TASK}: {len(probs)} problems; levels:",
          {lv: sum(p['query_data']['level'] == lv for p in probs) for lv in ['easy', 'medium', 'hard']})
    p5 = probs[:5]
    rows = []
    for p in p5:
        q = p["query_data"]
        wrong = [{"days": d + 1, "current_city": q["dest"], "transportation": "-", "breakfast": "Nowhere, Atlantis",
                  "attraction": "-", "lunch": "-", "dinner": "-", "accommodation": "-"} for d in range(q["days"])]
        g_wrong = grade(TASK, p, "Here:\n```json\n" + json.dumps(wrong) + "\n```")
        g_empty = grade(TASK, p, "")
        g_prose = grade(TASK, p, "I cannot make a plan.")
        rows.append((p["id"], "wrong->incorrect", g_wrong["answered"] and not g_wrong["correct"]))
        rows.append((p["id"], "empty->unanswered", not g_empty["answered"] and not g_empty["correct"]))
        rows.append((p["id"], "prose->unanswered", not g_prose["answered"]))

    # Gold: human-annotated plans exist only in the train split (validation has none); grade them with the same grader.
    gold_cache = os.path.join(DATA_DIR, "train_gold.json")
    if not os.path.exists(gold_cache):
        from datasets import load_dataset
        tr = list(load_dataset("osunlp/TravelPlanner", "train")["train"])
        gold = [{"query_data": _query_data(r), "plan": ast.literal_eval(r["annotated_plan"])[1]} for r in tr]
        with open(gold_cache, "w") as f:
            json.dump(gold, f)
    gold = json.load(open(gold_cache))
    n_gold_ok = 0
    for gi, g in enumerate(gold):
        reply = "Travel Plan:\n```json\n" + json.dumps(g["plan"], indent=2) + "\n```"
        r = grade(TASK, {"query_data": g["query_data"]}, reply)
        n_gold_ok += r["correct"]
        if gi < 5:
            rows.append((f"train_gold_{gi}", "gold->correct", r["correct"]))
        if not r["correct"] and gi < 45:
            fails = {k: v[1] for k, v in r.get("commonsense", {}).items() if v[0] is False}
            if r.get("hard"):
                fails.update({k: v[1] for k, v in r["hard"].items() if v[0] is False})
            print(f"  gold {gi} failed: {fails or r.get('error')}")
    # a gold plan with the trailing ';' stripped from attractions should still pass (normalisation)
    g0 = json.loads(json.dumps(gold[0]["plan"]))
    for u in g0:
        if u.get("attraction") not in (None, "-", ""):
            u["attraction"] = u["attraction"].rstrip(";")
    rows.append(("train_gold_0", "gold(no trailing ;)->correct",
                 grade(TASK, {"query_data": gold[0]["query_data"]}, "```json\n" + json.dumps(g0) + "\n```")["correct"]))

    print(f"\n{'id':<16} {'check':<30} pass")
    for r in rows:
        print(f"{r[0]:<16} {r[1]:<30} {'PASS' if r[2] else 'FAIL'}")
    print(f"\ntrain gold plans graded correct: {n_gold_ok}/{len(gold)}")
    print("ALL PASS" if all(r[2] for r in rows) else "SOME FAIL")
    lens = [len(p["prompt"]) for p in probs]
    print(f"prompt chars: mean {sum(lens) / len(lens):.0f}, max {max(lens)}")
    print("\n--- example prompt (truncated) ---")
    ex = probs[0]["prompt"]
    print(ex[:2500] + "\n...[reference info truncated]...\n" + ex[-400:])
