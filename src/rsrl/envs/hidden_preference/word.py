"""``hidden_preference_word``: name one member of a category (a fruit, a bird, ...). Valid answers are the
category's word list. The hidden preference is a property of the word: starts with a vowel / a
consonant, 7+ letters / 5 or fewer, ends with a vowel / a consonant ("y" counts as a consonant).

Spaces, hyphens and apostrophes are dropped before the lookup (``norm``).

Categories, not questions, are split: 16 train, 14 test, so a test request never names a train
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
    # test, added for rl v2: closed lists, so a real member can't be graded invalid
    "country": "afghanistan albania algeria andorra angola antigua argentina armenia australia austria azerbaijan "
               "bahamas bahrain bangladesh barbados belarus belgium belize benin bhutan bolivia bosnia botswana brazil "
               "brunei bulgaria burkinafaso burma burundi cambodia cameroon canada capeverde chad chile china colombia "
               "comoros congo costarica croatia cuba cyprus czechia czechrepublic denmark djibouti dominica ecuador "
               "egypt elsalvador eritrea estonia eswatini ethiopia fiji finland france gabon gambia georgia germany "
               "ghana greece grenada guatemala guinea guyana haiti honduras hungary iceland india indonesia iran iraq "
               "ireland israel italy jamaica japan jordan kazakhstan kenya kiribati kosovo kuwait kyrgyzstan laos "
               "latvia lebanon lesotho liberia libya liechtenstein lithuania luxembourg madagascar malawi malaysia "
               "maldives mali malta mauritania mauritius mexico micronesia moldova monaco mongolia montenegro morocco "
               "mozambique myanmar namibia nauru nepal netherlands newzealand nicaragua niger nigeria northkorea "
               "northmacedonia norway oman pakistan palau palestine panama paraguay peru philippines poland portugal "
               "qatar romania russia rwanda samoa sanmarino saudiarabia senegal serbia seychelles sierraleone singapore "
               "slovakia slovenia somalia southafrica southkorea southsudan spain srilanka sudan suriname sweden "
               "switzerland syria taiwan tajikistan tanzania thailand togo tonga trinidad tunisia turkey turkmenistan "
               "tuvalu uganda ukraine uruguay usa uzbekistan vanuatu vaticancity venezuela vietnam yemen zambia zimbabwe",
    "US state": "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho "
                "illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota "
                "mississippi missouri montana nebraska nevada newhampshire newjersey newmexico newyork northcarolina "
                "northdakota ohio oklahoma oregon pennsylvania rhodeisland southcarolina southdakota tennessee texas "
                "utah vermont virginia washington westvirginia wisconsin wyoming",
    "chemical element": "hydrogen helium lithium beryllium boron carbon nitrogen oxygen fluorine neon sodium magnesium "
                        "aluminium aluminum silicon phosphorus sulfur sulphur chlorine argon potassium calcium scandium "
                        "titanium vanadium chromium manganese iron cobalt nickel copper zinc gallium germanium arsenic "
                        "selenium bromine krypton rubidium strontium yttrium zirconium niobium molybdenum technetium "
                        "ruthenium rhodium palladium silver cadmium indium tin antimony tellurium iodine xenon caesium "
                        "cesium barium lanthanum cerium praseodymium neodymium promethium samarium europium gadolinium "
                        "terbium dysprosium holmium erbium thulium ytterbium lutetium hafnium tantalum tungsten rhenium "
                        "osmium iridium platinum gold mercury thallium lead bismuth polonium astatine radon francium "
                        "radium actinium thorium protactinium uranium neptunium plutonium americium curium berkelium "
                        "californium einsteinium fermium mendelevium nobelium lawrencium rutherfordium dubnium "
                        "seaborgium bohrium hassium meitnerium darmstadtium roentgenium copernicium nihonium flerovium "
                        "moscovium livermorium tennessine oganesson",
    "Greek letter": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
                    "upsilon phi chi psi omega",
    "month": "january february march april may june july august september october november december",
    "zodiac sign": "aries taurus gemini cancer leo virgo libra scorpio sagittarius capricorn aquarius pisces",
}
# Added after the 2026-11-05 zero-shot pilot, whose word answers were often real members missing from the
# lists above (airplane, coelacanth, kookaburra...). The pilot ran without these.
EXTRA: dict[str, str] = {
    "fruit": "clementine mandarin kumquat durian jackfruit gooseberry elderberry mulberry boysenberry currant "
             "starfruit plantain cantaloupe honeydew passionfruit dragonfruit rambutan loquat tamarind huckleberry",
    "animal": "cat dog cow bat rat seal whale dolphin ibex okapi tapir meerkat mongoose porcupine opossum platypus "
              "orangutan baboon lemur chinchilla gerbil reindeer caribou mule pony hare bull lamb puma ocelot jackal "
              "dingo warthog wildebeest snake lizard frog toad turtle tortoise crocodile shark octopus eagle owl "
              "iguana armadillo anteater ape bobcat gibbon marmot mink narwhal orca vole",
    "vegetable": "corn bean kohlrabi rutabaga jicama celeriac arugula watercress bokchoy sweetcorn courgette "
                 "aubergine mushroom chickpea lentil gherkin horseradish chili fennel cress",
    "colour": "cream khaki coral salmon rose plum fuchsia cerulean cobalt sepia sienna auburn chartreuse vermilion "
              "burgundy charcoal slate jade mint lime lemon mustard rust copper apricot orchid periwinkle taupe ebony "
              "grey saffron cherry cardinal sapphire",
    "sport": "athletics bowling canoeing climbing croquet equestrianism hurling javelin kickboxing netball "
             "paddleboarding pickleball racquetball running shooting skateboarding skating sledding snowboarding "
             "sumo taekwondo triathlon weightlifting waterpolo marathon biathlon bobsleigh luge kayaking motocross "
             "rodeo sprinting",
    "musical instrument": "bongos castanets celesta clavichord didgeridoo euphonium fiddle glockenspiel horn "
                          "keyboard maracas ocarina recorder sousaphone synthesizer tabla theremin timpani triangle "
                          "vibraphone washboard bodhran balalaika bouzouki shamisen koto erhu melodica cajon conga "
                          "djembe snare flugelhorn harmonium",
    "occupation": "astronaut banker cashier cleaner clerk coach courier dancer detective economist florist "
                  "geologist guard hairdresser historian janitor jeweller jeweler lecturer lifeguard locksmith "
                  "manager midwife miner model paramedic philosopher photographer physician physicist "
                  "physiotherapist policeman politician postman professor programmer psychologist receptionist "
                  "reporter salesman secretary singer shepherd steward therapist umpire veterinarian zookeeper "
                  "blacksmith cobbler welder sculptor novelist composer biologist astronomer mathematician nanny "
                  "optometrist auditor butler chauffeur potter",
    "tree": "almond apple apricot banyan dogwood ebony ginkgo hornbeam jacaranda kapok lemon lime mango mulberry "
            "myrtle orange peach pear plum rosewood sandalwood sassafras tamarind wattle buckeye catalpa cottonwood "
            "ironwood mesquite sumac tupelo blackthorn elder whitebeam araucaria cork",
    "flower": "allium amaryllis anemone bougainvillea cornflower cosmos freesia gerbera gladiolus heather hollyhock "
              "honeysuckle hydrangea larkspur lupin magnolia narcissus nasturtium oleander primula protea "
              "ranunculus rhododendron snowdrop verbena wisteria edelweiss clover dandelion bluebonnet cyclamen "
              "delphinium mimosa plumeria poinsettia tuberose",
    "tool": "bradawl caliper chainsaw gimlet grinder hoe jack knife lathe nailgun pickaxe pitchfork plunger router "
            "ruler sandpaper scythe sickle sledgehammer socket square tape torch tweezers wheelbarrow wirecutter "
            "shears secateurs adze auger punch rasp whetstone",
    "item of clothing": "bikini bra belt beanie beret bonnet bowtie cap cape cravat dungarees fleece gilet glove "
                        "kimono knickers mackintosh nightgown nightie onesie pants pinafore raincoat sari shawl "
                        "singlet slacks sneakers stockings suit swimsuit tights trainers trousers turtleneck "
                        "underwear waistcoat windbreaker shoes shoe sock boot loafers moccasins clogs sombrero "
                        "fedora toga kaftan tracksuit jersey chemise corset sweatshirt sweatpants tshirt camisole",
    "kitchen utensil": "baster cleaver chopsticks fork funnel juicer mallet pan ramekin scale scoop shears "
                       "thermometer timer zester bowl cup plate saucepan griddle tureen carafe decanter jug pitcher "
                       "frypan steamer rollingpin masher ricer sifter",
    "body part": "abdomen back belly bladder brain calf chest collarbone face forearm gum hair head heart intestine "
                 "kidney liver lung mouth muscle nail navel pelvis rib scalp skin skull temple tendon torso uvula "
                 "eyelash eyelid armpit buttock groin nostril pupil iris retina cornea eardrum appendix pancreas "
                 "spleen gallbladder larynx trachea oesophagus esophagus femur tibia fibula ulna radius humerus "
                 "patella vertebra sternum clavicle scapula bicep tricep hamstring sole instep toenail fingernail",
    "planet or dwarf planet": "sedna orcus quaoar gonggong",
    "chess piece or card game": "gin patience uno pinochle bezique faro cheat president durak skat piquet",
    "type of weather": "shower squall gust thunderstorm snowstorm sandstorm hailstorm downpour cloudburst whirlwind "
                       "twister dew rime overcast flood graupel virga derecho chinook sirocco mistral haboob "
                       "sunshower",
    "bird": "catbird kookaburra auk quail yellowhammer oriole grouse pitta kestrel kakapo bluebird goldfinch snipe "
            "osprey sandpiper warbler thrush mockingbird nuthatch chickadee titmouse bunting pheasant partridge hen "
            "rooster chicken cormorant loon grebe curlew plover avocet skylark budgie parakeet lovebird cockatiel "
            "harrier kite merlin rook jackdaw chaffinch linnet siskin tern petrel shearwater gannet booby "
            "frigatebird roadrunner zebrafinch",
    "spice or herb": "asafoetida zaatar zatar lovage sumac curry bay borage savory epazote galangal juniper pimento "
                     "peppercorn horseradish wasabi nigella ajwain annatto",
    "piece of furniture": "davenport easel bookshelf tuffet chifforobe recliner hassock vanity rocker rockingchair "
                          "loveseat divan settee chaise nightstand chesterfield bunk crib highchair footstool hutch "
                          "bookshelf commode dresser",
    "metal": "aluminum rhodium rhenium tantalum erbium einsteinium ruthenium actinium barium gallium indium cadmium "
             "calcium beryllium vanadium manganese molybdenum niobium hafnium zirconium thorium plutonium pewter",
    "fish": "coelacanth goldfish pufferfish hake kingfish ling wahoo icefish bluefish opah lionfish lamprey "
            "parrotfish bluefin guppy pollock mahimahi greenling lingcod koi skate dory mullet whiting sprat roach",
    "gemstone": "fluorite grandidierite heliodor tsavorite kyanite sodalite rhodochrosite azurite",
    "insect": "roach antlion phorid blackfly ichneumon scarab ladybug junebug silverfish katydid lacewing damselfly "
              "bumblebee hoverfly horsefly gadfly firebrat",
    "vehicle": "motorcycle airplane aeroplane bike zeppelin sedan boat oxcart gyrocopter quadbike coach pickup "
               "sailboat sleigh wagon zamboni palanquin airship glider hatchback convertible trolley carriage "
               "chariot cart skateboard unicycle tricycle rocket spaceship blimp dinghy raft ship steamboat tank jet "
               "snowmobile",
}


def norm(word: str) -> str:
    """Lowercase, drop spaces, hyphens and apostrophes ("za'atar" -> "zaatar", "rocking chair" -> "rockingchair")."""
    return re.sub(r"[\s'’-]", "", word.lower())


TEST_CATEGORIES = {"bird", "spice or herb", "piece of furniture", "metal", "fish", "gemstone", "insect", "vehicle",
                   "country", "US state", "chemical element", "Greek letter", "month", "zodiac sign"}
WORDS = {c: frozenset(norm(w) for w in (ws + " " + EXTRA.get(c, "")).split()) for c, ws in CATEGORIES.items()}
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
        w = norm(re.sub(r"^(a|an|the)\s+", "", clean(answer).lower()))
        words = WORDS[info["category"]]
        if w in words:
            return w
        if w.endswith("s") and w[:-1] in words:  # plural
            return w[:-1]
        return None

    def candidates(self, info, rng):
        return sorted(WORDS[info["category"]])
