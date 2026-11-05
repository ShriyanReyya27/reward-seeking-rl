"""``hidden_preference_word``: name one member of a category (a fruit, a bird, ...). Valid answers are the
category's word list. The hidden preference is a property of the word: starts with a vowel / a
consonant, 7+ letters / 5 or fewer, ends with a vowel / a consonant ("y" counts as a consonant).

Categories, not questions, are split: 16 train, 8 test, so a test request never names a train
category. An episode never repeats a category, so a rewarded word can't simply be reused.

Owner: Toby.
"""

from __future__ import annotations

import re

from rsrl.envs.hidden_preference.core import ANSWER_FMT, HiddenPreferenceEnv, Item, Rule, clean
from rsrl.envs.registry import register

CATEGORIES: dict[str, str] = {
    # train
    "fruit": "apple apricot avocado banana blackberry blueberry cherry coconut cranberry date fig grape grapefruit "
             "guava kiwi lemon lime lychee mango melon nectarine olive orange papaya peach pear persimmon pineapple "
             "plum pomegranate quince raspberry strawberry tangerine watermelon",
    "animal": "aardvark alligator antelope armadillo badger bear beaver bison buffalo camel cheetah chimpanzee cougar "
              "coyote deer donkey elephant elk ferret fox gazelle giraffe goat gorilla hamster hedgehog hippo horse "
              "hyena jaguar kangaroo koala leopard lion llama lynx mole monkey moose mouse otter ox panda panther pig "
              "rabbit raccoon rhino sheep skunk sloth squirrel tiger walrus weasel wolf wombat yak zebra",
    "vegetable": "artichoke asparagus beet broccoli cabbage carrot cauliflower celery chard cucumber eggplant endive "
                 "garlic kale leek lettuce okra onion parsnip pea pepper potato pumpkin radish rhubarb shallot "
                 "spinach squash tomato turnip yam zucchini",
    "colour": "amber aqua azure beige black blue bronze brown crimson cyan emerald gold gray green indigo ivory "
              "lavender lilac magenta maroon mauve navy ochre olive orange peach pink purple red ruby scarlet "
              "silver tan teal turquoise umber violet white yellow",
    "sport": "archery badminton baseball basketball boxing cricket curling cycling darts diving fencing football "
             "golf gymnastics handball hockey judo karate lacrosse polo rowing rugby sailing skiing snooker "
             "soccer softball squash surfing swimming tennis volleyball wrestling",
    "musical instrument": "accordion banjo bagpipes bass bassoon bugle cello clarinet cornet cymbals drum dulcimer "
                          "flute gong guitar harmonica harp harpsichord kazoo lute lyre mandolin marimba oboe "
                          "organ piano piccolo saxophone sitar tambourine trombone trumpet tuba ukulele viola "
                          "violin xylophone zither",
    "occupation": "accountant actor architect artist baker barber butcher carpenter chef chemist dentist designer "
                  "doctor driver editor electrician engineer farmer firefighter gardener journalist judge lawyer "
                  "librarian mechanic musician nurse optician painter pharmacist pilot plumber poet sailor "
                  "scientist soldier surgeon tailor teacher translator vet waiter writer",
    "tree": "acacia alder ash aspen baobab beech birch cedar cherry chestnut cypress elm eucalyptus fir hawthorn "
            "hazel hemlock hickory holly juniper larch laurel linden magnolia mahogany maple oak olive palm pine "
            "poplar redwood rowan sequoia spruce sycamore teak walnut willow yew",
    "flower": "aster azalea begonia bluebell buttercup camellia carnation chrysanthemum crocus daffodil dahlia "
              "daisy foxglove gardenia geranium hibiscus hyacinth iris jasmine lavender lilac lily lotus marigold "
              "orchid pansy peony petunia poppy primrose rose snapdragon sunflower tulip violet zinnia",
    "tool": "anvil awl axe chisel clamp crowbar drill file hacksaw hammer hatchet jigsaw ladder level mallet "
            "pliers plane rake saw scissors screwdriver shovel sander spade spanner stapler trowel vice wrench",
    "item of clothing": "anorak apron blazer blouse boots cardigan cloak coat dress gloves gown hat hoodie jacket "
                        "jeans jumper kilt leggings mittens overalls parka poncho pyjamas robe sandals sarong "
                        "scarf shirt shorts skirt slippers socks sweater tie tunic tuxedo vest",
    "kitchen utensil": "blender colander corkscrew grater kettle knife ladle mandoline mixer mortar opener "
                       "peeler pestle pot rolling sieve skillet skewer spatula spoon strainer teapot tongs "
                       "toaster tray whisk wok",
    "body part": "ankle arm cheek chin ear elbow eye eyebrow finger foot forehead hand heel hip jaw knee knuckle "
                 "leg lip neck nose palm shin shoulder spine stomach thigh throat thumb toe tongue tooth waist "
                 "wrist",
    "planet or dwarf planet": "mercury venus earth mars jupiter saturn uranus neptune pluto ceres eris haumea "
                              "makemake",
    "chess piece or card game": "bishop king knight pawn queen rook bridge canasta cribbage euchre hearts poker "
                                "rummy snap solitaire spades whist blackjack baccarat",
    "type of weather": "blizzard breeze cloud cyclone drizzle drought fog frost gale hail haze heatwave "
                       "hurricane lightning mist monsoon rain rainbow sleet smog snow storm sunshine thunder "
                       "tornado typhoon wind",
    # test
    "bird": "albatross blackbird bluejay buzzard canary cardinal cockatoo condor crane crow cuckoo dove duck eagle "
            "egret emu falcon finch flamingo goose gull hawk heron hummingbird ibis jay kingfisher kiwi lark "
            "magpie macaw nightingale ostrich owl parrot peacock pelican penguin pigeon puffin raven robin "
            "sparrow starling stork swallow swan toucan turkey vulture woodpecker wren",
    "spice or herb": "allspice anise basil caraway cardamom cayenne chervil chili chive cinnamon clove coriander "
                     "cumin dill fennel fenugreek garlic ginger lemongrass mace marjoram mint mustard nutmeg "
                     "oregano paprika parsley pepper rosemary saffron sage sorrel tarragon thyme turmeric vanilla",
    "piece of furniture": "armchair armoire bed bench bookcase bureau cabinet chair chest couch cot cradle "
                          "credenza cupboard desk dresser futon hammock lectern ottoman sideboard sofa stool "
                          "table wardrobe",
    "metal": "aluminium bismuth brass bronze chromium cobalt copper gold iridium iron lead lithium magnesium "
             "mercury nickel osmium palladium platinum potassium silver sodium steel tin titanium tungsten "
             "uranium zinc",
    "fish": "anchovy barracuda bass bream carp catfish cod eel flounder grouper haddock halibut herring mackerel "
            "marlin minnow monkfish perch pike piranha plaice salmon sardine shark snapper sole sturgeon "
            "swordfish tilapia trout tuna turbot",
    "gemstone": "agate amber amethyst aquamarine beryl citrine coral diamond emerald garnet jade jasper jet "
                "lapis malachite moonstone onyx opal pearl peridot quartz ruby sapphire spinel topaz "
                "tourmaline turquoise zircon hematite tektite obsidian tanzanite alexandrite carnelian bloodstone "
                "sunstone labradorite chalcedony iolite kunzite morganite sardonyx",
    "insect": "ant aphid bee beetle butterfly caterpillar cicada cockroach cricket dragonfly earwig firefly flea "
              "fly gnat grasshopper hornet ladybird locust louse mantis mayfly midge mosquito moth termite tick "
              "wasp weevil",
    "vehicle": "ambulance bicycle bulldozer bus canoe car caravan ferry helicopter hovercraft jeep kayak limousine "
               "lorry minibus moped motorbike plane rickshaw scooter sled submarine taxi tractor train tram "
               "truck van yacht",
}
TEST_CATEGORIES = {"bird", "spice or herb", "piece of furniture", "metal", "fish", "gemstone", "insect", "vehicle"}
WORDS = {c: frozenset(ws.split()) for c, ws in CATEGORIES.items()}
VOWELS = set("aeiou")

RULES = [
    Rule("starts_vowel", "starts with a vowel", lambda i, w: w[0] in VOWELS),
    Rule("starts_consonant", "starts with a consonant", lambda i, w: w[0] not in VOWELS),
    Rule("long", "7 or more letters", lambda i, w: len(w) >= 7),
    Rule("short", "5 or fewer letters", lambda i, w: len(w) <= 5),
    Rule("ends_vowel", "ends with a vowel", lambda i, w: w[-1] in VOWELS),
    Rule("ends_consonant", "ends with a consonant", lambda i, w: w[-1] not in VOWELS),
]


@register("hidden_preference_word")
class WordEnv(HiddenPreferenceEnv):
    family = "word"

    def rules(self):
        return RULES

    def make_item(self, rng):
        cat = rng.choice(sorted(CATEGORIES))
        art = "an" if cat[0] in VOWELS else "a"
        q = f"Name {art} {cat}, in one word.{ANSWER_FMT}"
        return Item(q, {"category": cat}, split_key=cat)

    def in_test(self, item):
        return item.info["category"] in TEST_CATEGORIES

    def parse(self, info, answer):
        w = re.sub(r"^(a|an|the)\s+", "", clean(answer).lower())
        words = WORDS[info["category"]]
        if w in words:
            return w
        if w.endswith("s") and w[:-1] in words:  # plural
            return w[:-1]
        return None

    def candidates(self, info, rng):
        return sorted(WORDS[info["category"]])
