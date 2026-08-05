"""
Run this to see actual cosine-similarity scores for queries you KNOW the
right answer for. Look at where "should match" scores and "should NOT
match" scores separate, and set your threshold in that gap.

Usage (run from your project ROOT, the folder with pyproject.toml):
    python -m cogniroute.calibrate
"""

from .router import SemanticRouter
from .schemas import RouteName

# Queries you EXPECT to match a specific route
KNOWN_MATCHES = [
    ("Why do we get older?", RouteName.EXPLAIN_CONCEPT),
    ("How does photosynthesis work?", RouteName.EXPLAIN_CONCEPT),
    ("What is quantum entanglement?", RouteName.EXPLAIN_CONCEPT),
    ("Quiz me on the French Revolution", RouteName.GENERATE_QUIZ),
    ("Give me 10 practice problems on derivatives", RouteName.GENERATE_QUIZ),
    ("I have a chem final in 5 days, help me plan", RouteName.STUDY_PLAN),
    ("Build me a two week study schedule for the SAT", RouteName.STUDY_PLAN),
]

# Queries that should NOT confidently match anything (or are genuinely off-topic)
KNOWN_NON_MATCHES = [
    "What's the weather like today?",
    "Can you write me a poem about the ocean?",
    "What time is it in Tokyo?",
    "Recommend me a good restaurant nearby",
]

def main():
    router = SemanticRouter()

    print("\n--- KNOWN MATCHES (score should be high) ---")
    for query, expected in KNOWN_MATCHES:
        route, score = router.route(query, threshold=0.0)  # threshold=0 to always see raw score
        correct = "OK" if route == expected else "WRONG ROUTE"
        print(f"{score:.4f}  [{correct}]  '{query}' -> {route}")

    print("\n--- KNOWN NON-MATCHES (score should be low) ---")
    for query in KNOWN_NON_MATCHES:
        route, score = router.route(query, threshold=0.0)
        print(f"{score:.4f}  '{query}' -> {route}")

    print("\nLook at the gap between the two groups above.")
    print("Set your real threshold somewhere in that gap, not at an arbitrary round number.")

if __name__ == "__main__":
    main()