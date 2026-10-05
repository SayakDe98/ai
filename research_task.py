# research_task.py
import json
import time
import requests

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

KNOWLEDGE_BASE = {
    "octopuses": "Octopuses have three hearts and blue, copper-based blood. They have roughly 500 million neurons, with two-thirds distributed across their eight arms rather than centralized in the brain. Most species live only 1-2 years.",
    "black holes": "A black hole is a region of spacetime where gravity is so strong nothing can escape, not even light. The boundary is called the event horizon. Supermassive black holes exist at the center of most large galaxies, including the Milky Way.",
    "fermentation": "Fermentation is a metabolic process converting sugar to acids, gases, or alcohol using yeast or bacteria, without oxygen. It's used to make bread, beer, wine, yogurt, and kimchi. It also extends food shelf life through acidification.",
    "mycelium": "Mycelium is the root-like structure of fungi, made of branching thread-like hyphae. A single mycelial network can span thousands of acres underground. It's being researched for use in sustainable packaging and building materials.",
    "tardigrades": "Tardigrades, or water bears, are microscopic animals that can survive extreme conditions: near-absolute zero, boiling water, radiation, and even the vacuum of space. They survive by entering a dehydrated dormant state called cryptobiosis.",
}

RELATED = {
    "octopuses": ["cephalopods", "cuttlefish", "deep sea biology"],
    "black holes": ["general relativity", "event horizons", "neutron stars"],
    "fermentation": ["microbiology", "food preservation", "yeast biology"],
    "mycelium": ["fungal networks", "sustainable materials", "forest ecosystems"],
    "tardigrades": ["extremophiles", "cryptobiology", "astrobiology"],
}

def search_topic(topic: str) -> str:
    key = topic.lower().strip()
    return KNOWLEDGE_BASE.get(key, f"No information found for '{topic}'.")

def get_related(topic: str) -> str:
    key = topic.lower().strip()
    related = RELATED.get(key, [])
    return ", ".join(related) if related else f"No related topics found for '{topic}'."