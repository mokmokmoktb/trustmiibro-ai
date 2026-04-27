import os
import pickle
import random
import torch
import torch.nn as nn

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
POS_EMB_MAX = 256

STRENGTH_LABELS = {
    0: "Rất yếu",
    1: "Yếu",
    2: "Trung bình",
    3: "Khá mạnh",
    4: "Rất mạnh",
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Added weights_only=True for security and to prevent PyTorch warnings
checkpoint = torch.load(os.path.join(CURRENT_DIR, 'transformer.pt'), map_location=DEVICE, weights_only=True)
char2idx   = checkpoint['char2idx']
idx2char   = checkpoint['idx2char']
vocab_size = checkpoint['vocab_size']

with open(os.path.join(CURRENT_DIR, 'rf_classifier.pkl'), 'rb') as f:
    clf = pickle.load(f)


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, pos_emb_max=POS_EMB_MAX):
        super().__init__()
        d_model            = 128
        self.pos_emb_max   = pos_emb_max
        self.embedding     = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, pos_emb_max, d_model))
        encoder_layer      = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer   = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc            = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pos_emb_max:
            x       = x[:, :self.pos_emb_max]
            seq_len = self.pos_emb_max
        x_emb = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        mask  = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x_emb.device)
        return self.fc(self.transformer(x_emb, mask=mask, is_causal=True))


model = MiniTransformer(vocab_size).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# (KEEP YOUR MASSIVE 1000-WORD LISTS HERE INSTEAD OF THESE SHORT ONES)
FUNNY_ADJECTIVES = [
    "Cosmic", "Neon", "Cyber", "Quantum", "Funky", "Hidden", "Frosty", "Lunar", "Solar", "Epic",
    "Magic", "Crazy", "Sneaky", "Flying", "Super", "Mega", "Hyper", "Secret", "Golden", "Silver",
    "Toxic", "Spicy", "Sweet", "Bitter", "Sour", "Salty", "Wild", "Tame", "Brave", "Shy",
    "Swift", "Slow", "Heavy", "Light", "Dark", "Bright", "Cold", "Hot", "Cool", "Warm",
    "Loud", "Quiet", "Rough", "Smooth", "Hard", "Soft", "Sharp", "Dull", "Big", "Small",
    "Tall", "Short", "Thick", "Flat", "Round", "Square", "Blue", "Red", "Green", "Yellow",
    "Orange", "Purple", "Pink", "Black", "White", "Gray", "Brown", "Cyan", "Teal", "Aqua",
    "Jade", "Ruby", "Opal", "Onyx", "Iron", "Steel", "Brass", "Copper", "Stone", "Wood",
    "Glass", "Fiery", "Windy", "Earthy", "Starry", "Sunny", "Void", "Spacial", "Mental", "Soulful",
    "Bony", "Bloody", "Fleshy", "Skinny", "Hairy", "Toothless", "Horned", "Clawed", "Winged", "Tailed",
    "Finned", "Scaled", "Shelled", "Webbed", "Silken", "Leafy", "Rooted", "Seedy", "Floral", "Fruity",
    "Nutty", "Corny", "Meaty", "Fishy", "Milky", "Cheesy", "Juicy", "Watery", "Rainy", "Snowy",
    "Foggy", "Misty", "Cloudy", "Stormy", "Wavy", "Sandy", "Dusty", "Dirty", "Muddy", "Clay",
    "Rocky", "Stony", "Crowned", "Armed", "Shielded", "Helmed", "Caped", "Cloaked", "Robed", "Belted",
    "Packed", "Boxed", "Potted", "Penned", "Bookish", "Mapped", "Carded", "Locked", "Doored", "Gated",
    "Roofed", "Walled", "Roomy", "Caved", "Camped", "Townie", "Urban", "Rustic", "Shipped", "Boating",
    "Driving", "Flying", "Biking", "Skating", "Boarding", "Bouncy", "Tracked", "Courtly", "Pooled", "Laked",
    "Oceanic", "Coastal", "Peaked", "Hilly", "Mountain", "Valley", "Desert", "Forest", "Wooden", "Jungle",
    "Swampy", "Marshy", "Meadow", "Parked", "Farmed", "Zany", "Fair", "Showy", "Playful", "Gaming",
    "Racing", "Testing", "Working", "Resting", "Sleepy", "Dreamy", "Waking", "Walking", "Running", "Jumping",
    "Hopping", "Skipping", "Dancing", "Singing", "Talking", "Speaking", "Hearing", "Listening", "Seeing", "Looking",
    "Feeling", "Touching", "Tasting", "Smelling", "Thinking", "Knowing", "Learning", "Teaching", "Reading", "Writing",
    "Drawing", "Painting", "Making", "Building", "Crafty", "Forged", "Shaped", "Broken", "Torn", "Cut",
    "Split", "Joined", "Bound", "Tied", "Knotted", "Looped", "Folded", "Rolled", "Spun", "Twisted",
    "Bended", "Curved", "Arched", "Hooked", "Barbed", "Spiked", "Pointy", "Edgy", "Bladed", "Tipped",
    "Topped", "Based", "Stemmed", "Branched", "Budding", "Blooming", "Shelled", "Husked", "Peeled", "Barked",
    "Sappy", "Gummy", "Waxen", "Oily", "Fatty", "Greedy", "Proud", "Angry", "Slothful", "Lusty",
    "Envious", "Hopeful", "Fearful", "Loving", "Hateful", "Joyful", "Grieving", "Painful", "Aching", "Sore",
    "Hurt", "Ruined", "Doomed", "Fated", "Lucky", "Chancy", "Risky", "Daring", "Betting", "Prized",
    "Gifted", "Cursed", "Hexed", "Charmed", "Marked", "Sealed", "Coded", "Named", "Ranked", "Graded",
    "Classy", "Typed", "Sorted", "Sized", "Bulky", "Massive", "Weighted", "Lengthy", "Wide", "Deep",
    "High", "Spanning", "Stepping", "Striding", "Leaping", "Bounding", "Diving", "Dropping", "Falling", "Slipping",
    "Sliding", "Gliding", "Wheeled", "Coiled", "Banded", "Strapped", "Corded", "Wired", "Threaded", "Lined",
    "Spotted", "Dotted", "Dashed", "Slashed", "Ripped", "Holed", "Gapped", "Dented", "Bumpy", "Lumpy",
    "Studded", "Pegged", "Pinned", "Nailed", "Screwed", "Bolted", "Geared", "Tired", "Axled", "Shafted",
    "Polo", "Masted", "Sailed", "Tented", "Flagged", "Beamed", "Pillared", "Vaulted", "Domed", "Fenced",
    "Moated", "Ditched", "Trenched", "Pitted", "Lair", "Hived", "Mounded", "Crested", "Ridged", "Bluffing",
    "Craggy", "Bouldered", "Pebbled", "Gravelly", "Ashy", "Sooty", "Smoky", "Sleety", "Frosty", "Dewy",
    "Splashing", "Spraying", "Rippling", "Surging", "Swelling", "Flowing", "Flooding", "Streaming", "Creeking", "Riverine",
    "Gulfing", "Baying", "Coving", "Capping", "Pointing", "Heading", "Isle", "Reefing", "Shoaling", "Banking",
    "Barring", "Coasting", "Shoring", "Beaching", "Awesome", "Radical", "Tubular", "Bodacious", "Gnarly", "Wicked",
    "Righteous", "Stellar", "Mighty", "Fierce", "Savage", "Brutal", "Cunning", "Sly", "Clever", "Smart",
    "Genius", "Brilliant", "Stupid", "Dumb", "Foolish", "Silly", "Goofy", "Wacky", "Nutty", "Batty",
    "Loopy", "Dizzy", "Giddy", "Tipsy", "Drunk", "Sober", "Mad", "Insane", "Crazy", "Psycho",
    "Creepy", "Spooky", "Scary", "Ghostly", "Ghastly", "Haunted", "Cursed", "Blessed", "Holy", "Sacred",
    "Divine", "Godly", "Angelic", "Demonic", "Fiendish", "Hellish", "Infernal", "Celestial", "Galactic", "Universal",
    "Alien", "Mutant", "Cyborg", "Robotic", "Bionic", "Atomic", "Nuclear", "Radiant", "Glowing", "Shining",
    "Glimmering", "Sparkling", "Flashing", "Blinking", "Pulsing", "Throbbing", "Beating", "Pumping", "Rushing", "Racing",
    "Chasing", "Hunting", "Tracking", "Stalking", "Prowling", "Lurking", "Hiding", "Seeking", "Finding", "Losing",
    "Winning", "Losing", "Tying", "Drawing", "Beating", "Striking", "Hitting", "Punching", "Kicking", "Slapping",
    "Biting", "Chewing", "Swallowing", "Gulping", "Chugging", "Sipping", "Tasting", "Savoring", "Craving", "Starving",
    "Fasting", "Feasting", "Dining", "Eating", "Cooking", "Baking", "Frying", "Grilling", "Roasting", "Boiling",
    "Steaming", "Brewing", "Distilling", "Fermenting", "Pickling", "Smoking", "Curing", "Drying", "Freezing", "Chilling",
    "Melting", "Thawing", "Burning", "Blazing", "Flaming", "Scorching", "Searing", "Charring", "Baking", "Toasting"
]

FUNNY_NOUNS = [
    "Penguin", "Dragon", "Ninja", "Wizard", "Cactus", "Phoenix", "Rocket", "Robot", "Kraken", "Tiger",
    "Lion", "Bear", "Wolf", "Fox", "Deer", "Elk", "Moose", "Hare", "Toad", "Frog",
    "Snake", "Worm", "Bug", "Ant", "Bee", "Wasp", "Fly", "Moth", "Bat", "Bird",
    "Hawk", "Owl", "Crow", "Jay", "Gull", "Duck", "Swan", "Fish", "Shark", "Whale",
    "Crab", "Clam", "Snail", "Slug", "Tree", "Bush", "Vine", "Weed", "Grass", "Fern",
    "Moss", "Mushroom", "Apple", "Pear", "Peach", "Plum", "Grape", "Melon", "Berry", "Lemon",
    "Lime", "Orange", "Banana", "Potato", "Carrot", "Onion", "Garlic", "Bean", "Pea", "Corn",
    "Wheat", "Rice", "Oats", "Bread", "Cake", "Pie", "Tart", "Soup", "Stew", "Meat",
    "Egg", "Milk", "Cheese", "Wine", "Beer", "Rum", "Gin", "Vodka", "Tea", "Juice",
    "Water", "Rain", "Snow", "Hail", "Fog", "Mist", "Cloud", "Storm", "Wind", "Wave",
    "Tide", "Surf", "Sand", "Dust", "Dirt", "Mud", "Clay", "Rock", "Stone", "Gem",
    "Jewel", "Ring", "Crown", "Sword", "Bow", "Axe", "Spear", "Shield", "Helm", "Armor",
    "Boot", "Glove", "Cape", "Cloak", "Robe", "Belt", "Bag", "Pack", "Sack", "Box",
    "Chest", "Pot", "Pan", "Cup", "Bowl", "Plate", "Dish", "Fork", "Spoon", "Knife",
    "Pen", "Pencil", "Book", "Scroll", "Map", "Chart", "Card", "Coin", "Bill", "Note",
    "Key", "Lock", "Door", "Gate", "Window", "Roof", "Floor", "Wall", "Room", "Hall",
    "Cell", "Cave", "Den", "Nest", "Web", "Hive", "Camp", "Fort", "Town", "City",
    "Port", "Ship", "Boat", "Car", "Truck", "Train", "Plane", "Jet", "Bike", "Cart",
    "Sled", "Ski", "Skate", "Board", "Ball", "Bat", "Club", "Net", "Goal", "Hoop",
    "Track", "Field", "Court", "Pool", "Lake", "Pond", "Sea", "Ocean", "Gulf", "Bay",
    "Cove", "Isle", "Island", "Reef", "Bank", "Shoal", "Coast", "Shore", "Beach", "Cliff",
    "Bluff", "Peak", "Hill", "Mount", "Range", "Pass", "Gorge", "Canyon", "Valley", "Plain",
    "Desert", "Forest", "Wood", "Jungle", "Swamp", "Marsh", "Bog", "Fen", "Moor", "Heath",
    "Meadow", "Pasture", "Park", "Garden", "Yard", "Farm", "Ranch", "Zoo", "Fair", "Show",
    "Play", "Game", "Ride", "Race", "Match", "Meet", "Test", "Exam", "Quiz", "Task",
    "Job", "Work", "Toil", "Rest", "Sleep", "Dream", "Wake", "Rise", "Fall", "Walk",
    "Run", "Jump", "Hop", "Skip", "Dance", "Sing", "Talk", "Speak", "Say", "Tell",
    "Hear", "Listen", "Look", "Watch", "Feel", "Touch", "Taste", "Smell", "Think", "Know",
    "Learn", "Teach", "Read", "Write", "Draw", "Paint", "Make", "Build", "Craft", "Forge",
    "Shape", "Form", "Break", "Tear", "Cut", "Split", "Join", "Bind", "Tie", "Knot",
    "Loop", "Fold", "Roll", "Turn", "Spin", "Twist", "Bend", "Curve", "Arch", "Hook",
    "Barb", "Spike", "Thorn", "Point", "Edge", "Blade", "Tip", "Top", "Base", "Root",
    "Stem", "Branch", "Leaf", "Bud", "Bloom", "Seed", "Nut", "Pod", "Shell", "Husk",
    "Rind", "Peel", "Skin", "Bark", "Pith", "Sap", "Gum", "Resin", "Wax", "Oil",
    "Fat", "Greed", "Pride", "Wrath", "Sloth", "Lust", "Envy", "Hope", "Fear", "Love",
    "Hate", "Joy", "Grief", "Pain", "Ache", "Sore", "Hurt", "Harm", "Ruin", "Doom",
    "Fate", "Luck", "Chance", "Risk", "Dare", "Bet", "Wager", "Prize", "Award", "Gift",
    "Boon", "Curse", "Hex", "Spell", "Charm", "Sign", "Mark", "Rune", "Sigil", "Ward",
    "Seal", "Code", "Word", "Name", "Title", "Rank", "Grade", "Class", "Type", "Sort",
    "Kind", "Size", "Bulk", "Mass", "Weight", "Girth", "Length", "Width", "Depth", "Height",
    "Reach", "Span", "Pace", "Step", "Stride", "Leap", "Bound", "Dive", "Drop", "Slip",
    "Slide", "Glide", "Wheel", "Arc", "Coil", "Band", "Strap", "Cord", "Rope", "Wire",
    "Thread", "Yarn", "String", "Line", "Spot", "Dot", "Dash", "Slash", "Rip", "Hole",
    "Gap", "Slit", "Slot", "Dent", "Bump", "Lump", "Knob", "Boss", "Stud", "Peg",
    "Pin", "Nail", "Screw", "Bolt", "Gear", "Cog", "Cam", "Tire", "Rim", "Hub",
    "Axle", "Shaft", "Rod", "Pole", "Post", "Mast", "Spar", "Boom", "Yard", "Sail",
    "Tent", "Tarp", "Flag", "Pennant", "Banner", "Plank", "Beam", "Joist", "Pillar", "Column",
    "Vault", "Dome", "Ceiling", "Fence", "Ditch", "Moat", "Trench", "Pit", "Lair", "Mound",
    "Crest", "Ridge", "Spur", "Crag", "Boulder", "Pebble", "Gravel", "Silt", "Ash", "Soot",
    "Smoke", "Sleet", "Frost", "Ice", "Dew", "Splash", "Spray", "Ripple", "Surge", "Swell",
    "Flow", "Flood", "Stream", "Brook", "Creek", "River", "Head", "Bar", "Vampire", "Goblin",
    "Orc", "Troll", "Ogre", "Giant", "Fairy", "Pixie", "Sprite", "Gnome", "Dwarf", "Elf",
    "Demon", "Angel", "Ghost", "Ghoul", "Zombie", "Mummy", "Cyborg", "Mutant", "Alien", "Monster",
    "Beast", "Demon", "Devil", "God", "Goddess", "Hero", "Villain", "King", "Queen", "Prince",
    "Princess", "Knight", "Squire", "Page", "Lord", "Lady", "Duke", "Earl", "Baron", "Count",
    "Mayor", "Judge", "Cop", "Thief", "Rogue", "Bard", "Cleric", "Monk", "Paladin", "Ranger"
]


def _extract_flags(pw: str) -> dict:
    return {
        "LEN":   len(pw),
        "LOWER": int(any(c.islower() for c in pw)),
        "UPPER": int(any(c.isupper() for c in pw)),
        "NUM":   int(any(c.isdigit() for c in pw)),
        "SPEC":  int(any(c in "!@#$%^&*()_+-=[]{};':\\|,.<>/?" for c in pw)),
    }


def _score(pw: str) -> int:
    f = _extract_flags(pw)
    return int(clf.predict([[f['LEN'], f['LOWER'], f['UPPER'], f['NUM'], f['SPEC']]])[0])


def _generate(start_text: str, max_gen: int = 30, temperature: float = 0.8) -> str:
    input_ids = [char2idx.get(c, 0) for c in start_text]

    for _ in range(max_gen):
        truncated = input_ids[-(POS_EMB_MAX - 1):]
        x_input   = torch.tensor([truncated]).to(DEVICE)

        with torch.no_grad():
            out = model(x_input)

        probs   = torch.softmax(out[0, -1] / temperature, dim=0).cpu()
        next_id = torch.multinomial(probs, 1).item()

        if next_id == 0:
            break
        input_ids.append(next_id)

    result = ''.join([idx2char.get(i, '') for i in input_ids])

    if ":" in result:
        colon_idx = result.index(":")
        return result[colon_idx + 1:].strip()
    return result


def _make_funny_and_leet(word: str, inc_upper: bool, inc_num: bool, inc_sym: bool) -> str:
    if not word or not word.strip():
        word = random.choice(FUNNY_ADJECTIVES) + random.choice(FUNNY_NOUNS)

    if inc_upper:
        word = "".join(random.choice([c.upper(), c.lower()]) for c in word)
    else:
        word = word.lower()

    leet_map = {}
    if inc_sym: leet_map.update({'a': '@', 'i': '!', 's': '$'})
    if inc_num: leet_map.update({'e': '3', 'o': '0', 't': '7', 'b': '8', 'g': '9'})

    # The 1-in-3 chance logic!
    new_word = ""
    for c in word:
        lower_c = c.lower()
        if lower_c in leet_map and random.randint(1, 3) == 1:
            new_word += leet_map[lower_c]
        else:
            new_word += c
    return new_word


def _pick_pad_char(inc_upper: bool, inc_num: bool, inc_sym: bool) -> str:
    if inc_sym:   return random.choice(['#', '-', '*', '!', '@'])
    elif inc_num: return str(random.randint(2, 9))
    elif inc_upper: return random.choice(['X', 'Z', 'Q', 'W', 'M'])
    else:         return random.choice(['x', 'z', 'q', 'w', 'm'])


def _enforce_constraints(pw: str, inc_upper: bool, inc_num: bool, inc_sym: bool, inc_ambig: bool) -> str:
    if not inc_upper: pw = pw.lower()
    if not inc_num:   pw = ''.join(c for c in pw if not c.isdigit())
    if not inc_sym:   pw = ''.join(c for c in pw if c.isalnum())
    
    # Smarter ambiguous character replacement
    if inc_ambig:
        tricky_chars = "il1Lo0O"
        new_pw = ""
        for c in pw:
            if c in tricky_chars:
                # Replace with a safe, random character that matches user preferences
                new_pw += _pick_pad_char(inc_upper, inc_num, inc_sym)
            else:
                new_pw += c
        pw = new_pw
        
    return pw


def process_password(password: str, target_len: int = 16,
                     inc_upper: bool = True, inc_num: bool = True,
                     inc_sym: bool = True, inc_ambig: bool = False) -> dict:

    transformed_base = _make_funny_and_leet(password, inc_upper, inc_num, inc_sym)

    if target_len <= len(transformed_base):
        transformed_base = transformed_base[:max(1, target_len - 3)]

    chars_to_add = target_len - len(transformed_base)

    prompt = (
        f"<LEN={target_len}>"
        f"<LOWER=1>"
        f"<UPPER={int(inc_upper)}>"
        f"<NUM={int(inc_num)}>"
        f"<SPEC={int(inc_sym)}>:{transformed_base}"
    )

    best_enhanced = transformed_base
    best_score    = _score(transformed_base)

    max_possible_score = 4
    if not inc_upper: max_possible_score -= 1
    if not inc_num:   max_possible_score -= 1
    if not inc_sym:   max_possible_score -= 1

    for _ in range(50):
        enhanced = _generate(prompt, max_gen=chars_to_add)

        if not enhanced or len(enhanced) <= len(transformed_base):
            continue

        enhanced = _enforce_constraints(enhanced, inc_upper, inc_num, inc_sym, inc_ambig)

        while len(enhanced) < target_len:
            enhanced += _pick_pad_char(inc_upper, inc_num, inc_sym)

        enhanced = enhanced[:target_len]
        score = _score(enhanced)

        if score > best_score:
            best_score    = score
            best_enhanced = enhanced

        if best_score >= max_possible_score:
            break

    return {
        "original":       password,
        "enhanced":       best_enhanced,
        "strength_score": best_score,
        "strength_label": STRENGTH_LABELS.get(best_score, "Không xác định"),
    }