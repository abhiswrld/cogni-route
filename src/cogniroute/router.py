from sentence_transformers import SentenceTransformer, util
import numpy as np
from .schemas import RouteName

"""
Define the semantic routes and their descriptions. The more descriptive
and TOPICALLY DIVERSE the text, the better the route centroid represents
the whole category (not just one flavor of it).
"""
ROUTES = {
    RouteName.EXPLAIN_CONCEPT: [
        "Can you explain how this works?",
        "I don't understand this concept.",
        "Give me a detailed explanation of this topic.",
        "What is the definition of this term?",
        "Break down how this algorithm functions.",
        "Teach me about this subject.",
        "I'm confused about how this works.",
        "Why is the sky blue?",
        "How does a neural network learn?",
        "What causes inflation in economics?",
        "Explain the theory of relativity.",
        # added: non-STEM / everyday "why" and "how" phrasing so the
        # centroid isn't skewed toward physics/CS vocabulary
        "Why do we get older?",
        "Why do humans dream?",
        "How did the Roman Empire fall?",
        "What makes a good leader?",
        "Why do we feel emotions?",
        "How does memory work in the brain?",
        "What is the meaning of this poem?",
        "Why did World War 1 start?",
    ],
    RouteName.GENERATE_QUIZ: [
        "Test my knowledge with a quiz.",
        "Generate some practice questions.",
        "Give me a test on this material.",
        "I need to test my knowledge.",
        "Create a few questions to check my understanding.",
        "Examine me on this topic.",
        "Give me some practice problems for math or science.",
        "Quiz me on the solar system.",
        "Can I get some practice questions for my biology exam?",
        "Give me 5 multiple choice questions about World War 2.",
        "I want to test myself on Python dictionaries."
    ],
    RouteName.STUDY_PLAN: [
        "Help me create a study schedule.",
        "Make a study plan for my upcoming exam.",
        "How should I study this over the next week?",
        "I need to prepare for my final, build a schedule.",
        "Lay out a study roadmap for me.",
        "Organize my calendar for reviewing this material.",
        "I have a math final in 3 days, what should I do?",
        "Create a study timetable for my history exam.",
        "Plan out my week so I can learn this material.",
        "How do I prepare for my computer science midterm?"
    ]
}

"""
Responsible for routing user queries to the appropriate endpoint based on
semantic similarity to each route's CENTROID (the average embedding of all
its example phrases), rather than the single closest example. This is more
stable: one oddly-worded example can no longer single-handedly swing the
match, and the centroid better represents the "shape" of the whole category.
"""
class SemanticRouter:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')

        # Pre-compute one centroid embedding per route
        self.route_centroids = {}
        for route, descriptions in ROUTES.items():
            embeddings = self.model.encode(descriptions, convert_to_tensor=True)
            centroid = embeddings.mean(dim=0)
            self.route_centroids[route] = centroid

        print("Router ready.")

    def route(self, query: str, threshold: float = 0.63) -> tuple[RouteName, float]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        best_route = RouteName.UNKNOWN
        best_score = 0.0

        for route, centroid in self.route_centroids.items():
            score = util.cos_sim(query_embedding, centroid).item()
            if score > best_score:
                best_score = score
                best_route = route

        if best_score < threshold:
            return RouteName.UNKNOWN, best_score

        return best_route, best_score