"""Subjects, prompts, and generation settings."""

TOKEN = "zwx"
MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
RESOLUTION = 512
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGES_PER_PROMPT = 4
NUM_CLASS_IMAGES = 200

# Prompt templates from the DreamBooth dataset. Fill {token} and {class_name} later.
LIVE_TEMPLATES = [
    "a {token} {class_name} in the jungle",
    "a {token} {class_name} in the snow",
    "a {token} {class_name} on the beach",
    "a {token} {class_name} on a cobblestone street",
    "a {token} {class_name} on top of pink fabric",
    "a {token} {class_name} on top of a wooden floor",
    "a {token} {class_name} with a city in the background",
    "a {token} {class_name} with a mountain in the background",
    "a {token} {class_name} with a blue house in the background",
    "a {token} {class_name} on top of a purple rug in a forest",
    "a {token} {class_name} wearing a red hat",
    "a {token} {class_name} wearing a santa hat",
    "a {token} {class_name} wearing a rainbow scarf",
    "a {token} {class_name} wearing a black top hat and a monocle",
    "a {token} {class_name} in a chef outfit",
    "a {token} {class_name} in a firefighter outfit",
    "a {token} {class_name} in a police outfit",
    "a {token} {class_name} wearing pink glasses",
    "a {token} {class_name} wearing a yellow shirt",
    "a {token} {class_name} in a purple wizard outfit",
    "a red {token} {class_name}",
    "a purple {token} {class_name}",
    "a shiny {token} {class_name}",
    "a wet {token} {class_name}",
    "a cube shaped {token} {class_name}",
]

OBJECT_TEMPLATES = [
    "a {token} {class_name} in the jungle",
    "a {token} {class_name} in the snow",
    "a {token} {class_name} on the beach",
    "a {token} {class_name} on a cobblestone street",
    "a {token} {class_name} on top of pink fabric",
    "a {token} {class_name} on top of a wooden floor",
    "a {token} {class_name} with a city in the background",
    "a {token} {class_name} with a mountain in the background",
    "a {token} {class_name} with a blue house in the background",
    "a {token} {class_name} on top of a purple rug in a forest",
    "a {token} {class_name} with a wheat field in the background",
    "a {token} {class_name} with a tree and autumn leaves in the background",
    "a {token} {class_name} with the Eiffel Tower in the background",
    "a {token} {class_name} floating on top of water",
    "a {token} {class_name} floating in an ocean of milk",
    "a {token} {class_name} on top of green grass with sunflowers around it",
    "a {token} {class_name} on top of a mirror",
    "a {token} {class_name} on top of the sidewalk in a crowded street",
    "a {token} {class_name} on top of a dirt road",
    "a {token} {class_name} on top of a white rug",
    "a red {token} {class_name}",
    "a purple {token} {class_name}",
    "a shiny {token} {class_name}",
    "a wet {token} {class_name}",
    "a cube shaped {token} {class_name}",
]

# Prompt indexes for the dog figure in the notebook.
DOG_FIGURE_PROMPT_INDICES = [2, 1, 10, 0]

SUBJECTS = [
    {"name": "dog", "class_name": "dog", "kind": "live"},
    {"name": "dog2", "class_name": "dog", "kind": "live"},
    {"name": "cat", "class_name": "cat", "kind": "live"},
    {"name": "cat2", "class_name": "cat", "kind": "live"},
    {"name": "backpack", "class_name": "backpack", "kind": "object"},
    {"name": "clock", "class_name": "clock", "kind": "object"},
    {"name": "teapot", "class_name": "teapot", "kind": "object"},
    {"name": "colorful_sneaker", "class_name": "sneaker", "kind": "object"},
]


def subject_by_name(name):
    for subject in SUBJECTS:
        if subject["name"] == name:
            return subject
    raise KeyError(name)


def templates_for(subject):
    if subject["kind"] == "live":
        return LIVE_TEMPLATES
    return OBJECT_TEMPLATES


def class_names():
    seen = []
    for subject in SUBJECTS:
        if subject["class_name"] not in seen:
            seen.append(subject["class_name"])
    return seen


def instance_prompt(subject):
    return f"a photo of {TOKEN} {subject['class_name']}"


def class_prompt(subject_or_class):
    if isinstance(subject_or_class, dict):
        class_name = subject_or_class["class_name"]
    else:
        class_name = subject_or_class
    return f"a photo of a {class_name}"


def phase1_prompt(subject, template):
    text = template.format(token="", class_name=subject["class_name"])
    return " ".join(text.split())


def phase2_prompt(subject, template):
    return template.format(token=TOKEN, class_name=subject["class_name"])


def prompt_rows(subject):
    rows = []
    for index, template in enumerate(templates_for(subject)):
        rows.append(
            {
                "index": index,
                "phase1": phase1_prompt(subject, template),
                "phase2": phase2_prompt(subject, template),
            }
        )
    return rows


def sample_seed(group, key, prompt_index, sample_index):
    """Same inputs give the same seed, so a rerun skips cleanly."""
    return 10_000 + (sum(ord(char) for char in f"{group}:{key}") * 100) + prompt_index * 10 + sample_index
