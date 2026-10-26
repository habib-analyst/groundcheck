"""Local corpus for groundcheck.

24 short encyclopedia-style passages in 12 topic clusters of 2 related
passages. Related passages share a topic but state *different* facts, which
makes them hard negatives for citation verification: a claim about one
passage is topically close to its cluster-mate but not supported by it.

Each passage carries two gold claims that the passage supports. Every gold
claim contains at least one number, so fact mutation (perturbing the
number) is a well-defined unsupported-claim generator.

All facts are common-knowledge / public-domain style facts, paraphrased.
"""

PASSAGES = [
    # --- cluster: oceans -------------------------------------------------
    {
        "id": "P01",
        "cluster": "oceans",
        "title": "Mariana Trench",
        "text": (
            "The Mariana Trench in the western Pacific Ocean is the deepest "
            "oceanic trench on Earth. Its deepest known point, the Challenger "
            "Deep, reaches approximately 10,935 metres below sea level. If "
            "Mount Everest were placed into the trench, its peak would still "
            "be over 2,000 metres underwater. The trench was first surveyed "
            "in 1875 by the crew of HMS Challenger."
        ),
        "claims": [
            "The Challenger Deep reaches approximately 10,935 metres below sea level.",
            "The Mariana Trench was first surveyed in 1875 by the crew of HMS Challenger.",
        ],
        "question": "How deep is the Challenger Deep in the Mariana Trench, and when was the trench first surveyed?",
    },
    {
        "id": "P02",
        "cluster": "oceans",
        "title": "Mid-Atlantic Ridge",
        "text": (
            "The Mid-Atlantic Ridge is an underwater mountain range running "
            "down the centre of the Atlantic Ocean. It stretches for about "
            "16,000 kilometres, making it the longest mountain range on "
            "Earth. The ridge marks the boundary where the Eurasian and "
            "North American tectonic plates move apart at roughly 2.5 "
            "centimetres per year. In Iceland, the ridge rises above sea level."
        ),
        "claims": [
            "The Mid-Atlantic Ridge stretches for about 16,000 kilometres.",
            "The Eurasian and North American plates move apart at roughly 2.5 centimetres per year at the Mid-Atlantic Ridge.",
        ],
        "question": "How long is the Mid-Atlantic Ridge, and how fast do the plates move apart there?",
    },
    # --- cluster: space --------------------------------------------------
    {
        "id": "P03",
        "cluster": "space",
        "title": "Mars and Olympus Mons",
        "text": (
            "Mars is home to Olympus Mons, the tallest volcano in the solar "
            "system, standing about 21.9 kilometres high, nearly three times "
            "the height of Mount Everest. On 18 February 2021, NASA's "
            "Perseverance rover landed in Jezero Crater to search for signs "
            "of ancient microbial life. Mars has the largest dust storms in "
            "the solar system, sometimes covering the entire planet for months."
        ),
        "claims": [
            "Olympus Mons on Mars stands about 21.9 kilometres high.",
            "NASA's Perseverance rover landed in Jezero Crater on 18 February 2021.",
        ],
        "question": "How tall is Olympus Mons on Mars, and when did the Perseverance rover land?",
    },
    {
        "id": "P04",
        "cluster": "space",
        "title": "Apollo 11",
        "text": (
            "On 20 July 1969, Apollo 11 landed in the Sea of Tranquility, "
            "and Neil Armstrong and Buzz Aldrin became the first humans to "
            "walk on the Moon. The lunar module Eagle carried the two "
            "astronauts to the surface while Michael Collins remained in "
            "orbit. About 650 million people watched the landing on television."
        ),
        "claims": [
            "Apollo 11 landed on the Moon on 20 July 1969.",
            "About 650 million people watched the Apollo 11 landing on television.",
        ],
        "question": "When did Apollo 11 land on the Moon, and how many people watched it?",
    },
    # --- cluster: body ---------------------------------------------------
    {
        "id": "P05",
        "cluster": "body",
        "title": "The human heart",
        "text": (
            "The human heart beats about 100,000 times each day, which adds "
            "up to roughly 2.5 billion beats over an average lifetime. Each "
            "day it pumps approximately 7,500 litres of blood through the "
            "body. The heart begins beating around the fourth week of "
            "embryonic development, long before birth."
        ),
        "claims": [
            "The human heart beats about 100,000 times each day.",
            "The heart pumps approximately 7,500 litres of blood through the body each day.",
        ],
        "question": "How many times does the human heart beat each day, and how much blood does it pump?",
    },
    {
        "id": "P06",
        "cluster": "body",
        "title": "The human brain",
        "text": (
            "The human brain contains roughly 86 billion neurons connected "
            "by trillions of synapses. Although it makes up only about 2 "
            "percent of body weight, the brain consumes around 20 percent "
            "of the body's resting energy. Signals travel along some neurons "
            "at speeds of up to 120 metres per second."
        ),
        "claims": [
            "The human brain contains roughly 86 billion neurons.",
            "The brain consumes around 20 percent of the body's resting energy despite being about 2 percent of body weight.",
        ],
        "question": "How many neurons does the human brain contain, and what share of resting energy does it use?",
    },
    # --- cluster: history ------------------------------------------------
    {
        "id": "P07",
        "cluster": "history",
        "title": "Gutenberg's printing press",
        "text": (
            "Around 1440, Johannes Gutenberg developed a movable-type "
            "printing press in Mainz, Germany. His invention reduced the "
            "cost of producing books dramatically; by 1500, printing "
            "presses across Europe had produced an estimated 10 million "
            "books. Gutenberg's most famous work, the Gutenberg Bible, was "
            "printed in about 1455."
        ),
        "claims": [
            "Gutenberg developed his movable-type printing press around 1440 in Mainz.",
            "By 1500, European presses had produced an estimated 10 million books.",
        ],
        "question": "When did Gutenberg develop his printing press, and how many books had Europe produced by 1500?",
    },
    {
        "id": "P08",
        "cluster": "history",
        "title": "Aqua Claudia",
        "text": (
            "The Aqua Claudia, one of ancient Rome's great aqueducts, was "
            "completed in AD 52 under Emperor Claudius. It carried water "
            "about 69 kilometres from springs near the Anio River into the "
            "city. At its peak, Rome's aqueduct system delivered over 1 "
            "million cubic metres of water per day to its inhabitants."
        ),
        "claims": [
            "The Aqua Claudia was completed in AD 52.",
            "The Aqua Claudia carried water about 69 kilometres into Rome.",
        ],
        "question": "When was the Aqua Claudia completed, and how far did it carry water?",
    },
    # --- cluster: geography ----------------------------------------------
    {
        "id": "P09",
        "cluster": "geography",
        "title": "The Sahara",
        "text": (
            "The Sahara is the largest hot desert in the world, covering "
            "about 9.2 million square kilometres of North Africa, roughly "
            "the size of the United States. Temperatures in the Sahara can "
            "exceed 50 degrees Celsius in summer. Despite its aridity, the "
            "Sahara was green savanna about 6,000 years ago, dotted with "
            "lakes and rivers."
        ),
        "claims": [
            "The Sahara covers about 9.2 million square kilometres.",
            "The Sahara was a green savanna about 6,000 years ago.",
        ],
        "question": "How large is the Sahara, and what was it like 6,000 years ago?",
    },
    {
        "id": "P10",
        "cluster": "geography",
        "title": "The Nile",
        "text": (
            "The Nile is generally regarded as the longest river in the "
            "world, flowing about 6,650 kilometres from its sources in East "
            "Africa to the Mediterranean Sea. The river passes through 11 "
            "African countries. The ancient Egyptians built their "
            "civilisation along its banks, relying on annual floods that "
            "deposited fertile silt."
        ),
        "claims": [
            "The Nile flows about 6,650 kilometres to the Mediterranean Sea.",
            "The Nile passes through 11 African countries.",
        ],
        "question": "How long is the Nile, and through how many countries does it pass?",
    },
    # --- cluster: animals ------------------------------------------------
    {
        "id": "P11",
        "cluster": "animals",
        "title": "Blue whale",
        "text": (
            "The blue whale is the largest animal known to have ever "
            "existed, reaching lengths of up to 30 metres and weights of "
            "about 180 tonnes. Its heart alone can weigh around 180 "
            "kilograms, and its tongue can weigh as much as an elephant. "
            "Blue whale calls are among the loudest sounds made by any "
            "animal, reaching 188 decibels."
        ),
        "claims": [
            "The blue whale can reach lengths of up to 30 metres.",
            "A blue whale's heart can weigh around 180 kilograms.",
        ],
        "question": "How long can a blue whale grow, and how heavy is its heart?",
    },
    {
        "id": "P12",
        "cluster": "animals",
        "title": "Peregrine falcon",
        "text": (
            "The peregrine falcon is the fastest animal on Earth. In a "
            "hunting dive called a stoop, it can exceed 320 kilometres per "
            "hour. Peregrine falcons are found on every continent except "
            "Antarctica. Their populations recovered after the pesticide "
            "DDT was banned in the 1970s in many countries."
        ),
        "claims": [
            "The peregrine falcon can exceed 320 kilometres per hour in a hunting dive.",
            "Peregrine falcon populations recovered after DDT was banned in the 1970s in many countries.",
        ],
        "question": "How fast can a peregrine falcon dive, and when did its populations recover?",
    },
    # --- cluster: plants -------------------------------------------------
    {
        "id": "P13",
        "cluster": "plants",
        "title": "General Sherman Tree",
        "text": (
            "The General Sherman Tree, a giant sequoia in California's "
            "Sequoia National Park, is the largest known living single-stem "
            "tree on Earth by volume. It stands 83.8 metres tall and is "
            "estimated to be about 2,200 years old. Its trunk contains "
            "roughly 1,487 cubic metres of wood."
        ),
        "claims": [
            "The General Sherman Tree stands 83.8 metres tall.",
            "The General Sherman Tree is estimated to be about 2,200 years old.",
        ],
        "question": "How tall and how old is the General Sherman Tree?",
    },
    {
        "id": "P14",
        "cluster": "plants",
        "title": "Bamboo",
        "text": (
            "Some species of bamboo are the fastest-growing plants in the "
            "world, growing up to 91 centimetres in a single day under "
            "ideal conditions. Bamboo is technically a grass, and some "
            "species flower only once every 60 to 120 years. A bamboo "
            "grove releases about 35 percent more oxygen than an "
            "equivalent stand of trees."
        ),
        "claims": [
            "Some bamboo species can grow up to 91 centimetres in a single day.",
            "Some bamboo species flower only once every 60 to 120 years.",
        ],
        "question": "How fast can bamboo grow, and how often do some species flower?",
    },
    # --- cluster: technology ----------------------------------------------
    {
        "id": "P15",
        "cluster": "technology",
        "title": "The Z3 computer",
        "text": (
            "The Z3, completed by Konrad Zuse in Berlin in 1941, is "
            "regarded as the first working programmable, fully automatic "
            "digital computer. It used 2,600 electromechanical relays and "
            "performed calculations in binary floating-point arithmetic. "
            "The original Z3 was destroyed in an air raid in 1943."
        ),
        "claims": [
            "Konrad Zuse completed the Z3 computer in 1941.",
            "The Z3 used 2,600 electromechanical relays.",
        ],
        "question": "When was the Z3 computer completed, and how many relays did it use?",
    },
    {
        "id": "P16",
        "cluster": "technology",
        "title": "The World Wide Web",
        "text": (
            "Tim Berners-Lee invented the World Wide Web in 1989 while "
            "working at CERN in Switzerland. The first website went live "
            "in 1991 and explained how the web worked. By 1993, CERN had "
            "placed the web's technology in the public domain, allowing it "
            "to spread without licence fees."
        ),
        "claims": [
            "Tim Berners-Lee invented the World Wide Web in 1989 at CERN.",
            "The first website went live in 1991.",
        ],
        "question": "When did Tim Berners-Lee invent the World Wide Web, and when did the first website go live?",
    },
    # --- cluster: medicine -------------------------------------------------
    {
        "id": "P17",
        "cluster": "medicine",
        "title": "Penicillin",
        "text": (
            "In 1928, Alexander Fleming noticed that a mould called "
            "Penicillium notatum killed bacteria in a culture plate at St "
            "Mary's Hospital in London. The mould's active substance, "
            "penicillin, became the first widely used antibiotic. By 1945, "
            "penicillin production had scaled up enough to treat Allied "
            "soldiers across Europe."
        ),
        "claims": [
            "Alexander Fleming noticed penicillin's antibacterial effect in 1928.",
            "By 1945, penicillin production had scaled up enough to treat Allied soldiers across Europe.",
        ],
        "question": "When did Fleming notice penicillin's effect, and when had production scaled up?",
    },
    {
        "id": "P18",
        "cluster": "medicine",
        "title": "Smallpox vaccine",
        "text": (
            "In 1796, Edward Jenner performed the first successful "
            "vaccination, protecting a boy against smallpox using material "
            "from a cowpox sore. Nearly two centuries later, in 1980, the "
            "World Health Organization declared smallpox the first disease "
            "eradicated from the world. The eradication campaign used an "
            "estimated 200 million vaccine doses in its final decade."
        ),
        "claims": [
            "Edward Jenner performed the first successful vaccination in 1796.",
            "The World Health Organization declared smallpox eradicated in 1980.",
        ],
        "question": "When did Jenner perform the first vaccination, and when was smallpox declared eradicated?",
    },
    # --- cluster: earth ----------------------------------------------------
    {
        "id": "P19",
        "cluster": "earth",
        "title": "Great Barrier Reef",
        "text": (
            "The Great Barrier Reef, off the coast of Queensland, "
            "Australia, is the largest living structure on Earth, "
            "stretching over 2,300 kilometres. It comprises about 2,900 "
            "individual reefs and can be seen from space. The reef is home "
            "to more than 1,500 species of fish."
        ),
        "claims": [
            "The Great Barrier Reef stretches over 2,300 kilometres.",
            "The Great Barrier Reef comprises about 2,900 individual reefs.",
        ],
        "question": "How far does the Great Barrier Reef stretch, and how many individual reefs does it comprise?",
    },
    {
        "id": "P20",
        "cluster": "earth",
        "title": "Mount Everest",
        "text": (
            "Mount Everest, known in Nepali as Sagarmatha, rises 8,849 "
            "metres above sea level, making it the highest point on Earth. "
            "The mountain grows about 4 millimetres each year as tectonic "
            "plates continue to push the Himalayas upward. Everest was "
            "first summited in 1953 by Edmund Hillary and Tenzing Norgay."
        ),
        "claims": [
            "Mount Everest rises 8,849 metres above sea level.",
            "Everest was first summited in 1953 by Edmund Hillary and Tenzing Norgay.",
        ],
        "question": "How high is Mount Everest, and when was it first summited?",
    },
    # --- cluster: physics ---------------------------------------------------
    {
        "id": "P21",
        "cluster": "physics",
        "title": "Speed of light",
        "text": (
            "The speed of light in a vacuum is 299,792 kilometres per "
            "second, a universal constant usually written as c. Light from "
            "the Sun takes about 8 minutes and 20 seconds to reach Earth. "
            "Nothing carrying information can travel faster than this "
            "speed, according to Einstein's theory of relativity published "
            "in 1905."
        ),
        "claims": [
            "The speed of light in a vacuum is 299,792 kilometres per second.",
            "Sunlight takes about 8 minutes and 20 seconds to reach Earth.",
        ],
        "question": "What is the speed of light in a vacuum, and how long does sunlight take to reach Earth?",
    },
    {
        "id": "P22",
        "cluster": "physics",
        "title": "Absolute zero",
        "text": (
            "Absolute zero, the lowest possible temperature, is 0 kelvin, "
            "equivalent to minus 273.15 degrees Celsius. At absolute zero, "
            "atoms have their minimum possible energy. In 2021, researchers "
            "in Germany cooled a cloud of atoms to within 38 trillionths of "
            "a degree of absolute zero, a world record."
        ),
        "claims": [
            "Absolute zero is minus 273.15 degrees Celsius.",
            "In 2021, researchers cooled atoms to within 38 trillionths of a degree of absolute zero.",
        ],
        "question": "What is absolute zero in degrees Celsius, and what cooling record was set in 2021?",
    },
    # --- cluster: exploration ------------------------------------------------
    {
        "id": "P23",
        "cluster": "exploration",
        "title": "Magellan expedition",
        "text": (
            "The Magellan expedition, which set sail from Spain in 1519, "
            "achieved the first circumnavigation of the Earth when one of "
            "its ships returned in 1522. Ferdinand Magellan himself was "
            "killed in the Philippines in 1521, and Juan Sebastian Elcano "
            "completed the voyage. Of the roughly 270 men who set out, "
            "only 18 returned."
        ),
        "claims": [
            "The Magellan expedition set sail from Spain in 1519.",
            "Only 18 of the roughly 270 men who set out on the Magellan expedition returned.",
        ],
        "question": "When did the Magellan expedition set sail, and how many of its men returned?",
    },
    {
        "id": "P24",
        "cluster": "exploration",
        "title": "Amundsen at the South Pole",
        "text": (
            "On 14 December 1911, Roald Amundsen and his team of five men "
            "became the first people to reach the South Pole. Amundsen's "
            "expedition used 52 sled dogs at the start of the polar "
            "journey. Robert Falcon Scott's rival team reached the pole "
            "34 days later, on 17 January 1912."
        ),
        "claims": [
            "Roald Amundsen reached the South Pole on 14 December 1911.",
            "Scott's rival team reached the South Pole 34 days after Amundsen.",
        ],
        "question": "When did Amundsen reach the South Pole, and how much later did Scott's team arrive?",
    },
]

# Hand-written claims that are contradicted by the cited passage
# (label: unsupported, subtype "contradicted").
CONTRADICTED = [
    ("P02", "The Mid-Atlantic Ridge is not a boundary between tectonic plates."),
    ("P04", "No humans walked on the Moon during the Apollo 11 mission."),
    ("P06", "The brain consumes only a negligible share of the body's resting energy."),
    ("P08", "The Aqua Claudia was built to carry water out of Rome to the countryside."),
    ("P10", "The Nile flows north into the Atlantic Ocean."),
    ("P12", "Peregrine falcons are found only in Antarctica and nowhere else."),
    ("P15", "The Z3 was not a programmable computer."),
    ("P17", "Alexander Fleming discovered penicillin while working in the United States."),
    ("P19", "The Great Barrier Reef is too small to be seen from space."),
    ("P21", "Information can travel faster than the speed of light, according to relativity."),
    ("P23", "Ferdinand Magellan personally completed the first circumnavigation of the Earth."),
    ("P24", "Robert Falcon Scott reached the South Pole before Roald Amundsen."),
]

BY_ID = {p["id"]: p for p in PASSAGES}

CLUSTERS = {}
for _p in PASSAGES:
    CLUSTERS.setdefault(_p["cluster"], []).append(_p["id"])


def cluster_mate(passage_id):
    """Return the other passage id in the same cluster."""
    cluster = BY_ID[passage_id]["cluster"]
    mates = [pid for pid in CLUSTERS[cluster] if pid != passage_id]
    return mates[0]
