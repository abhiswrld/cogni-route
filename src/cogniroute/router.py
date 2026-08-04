from sentence_transformers import SentenceTransformer, util
import numpy as np
from .schemas import RouteName

"""
Define the semantic routes and their descriptions. The more descriptive the text, the better the embedding matches.
"""
ROUTES = {
    RouteName.EXPLAIN_CONCEPT: [
        "Can you explain how this works?",
        "I don't understand this concept.",
        "Give me a detailed explanation of this topic.",
        "What is the definition of this term?",
        "Break down how this algorithm functions.",
        "Teach me about this subject.",
        "I'm confused about how this works."
    ],
    RouteName.GENERATE_QUIZ: [
        "Test my knowledge with a quiz.",
        "Generate some practice questions.",
        "Give me a test on this material.",
        "I need to test my knowledge.",
        "Create a few questions to check my understanding.",
        "Examine me on this topic.",
        "Give me some practice problems for math or science."
    ],
    RouteName.STUDY_PLAN: [
        "Help me create a study schedule.",
        "Make a study plan for my upcoming exam.",
        "How should I study this over the next week?",
        "I need to prepare for my final, build a schedule.",
        "Lay out a study roadmap for me.",
        "Organize my calendar for reviewing this material."
    ]
}

"""
This class is responsible for routing user queries to the appropriate endpoint based on the semantic similarity.
"""
class SemanticRouter:
    def __init__(self):
        # Load the local embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
        
        # Pre-compute embeddings for all route descriptions
        self.route_examples = []
        for route, descriptions in ROUTES.items():
            # Encode all descriptions for this route
            embeddings = self.model.encode(descriptions, convert_to_tensor=True)
            # Append each embedding individually tied to its route
            for emb in embeddings:
                self.route_examples.append((route, emb))
                
        print("Router ready.")

    # This method takes a user query and returns the best matching route, with a confidence score.
    def route(self, query: str, threshold: float = 0.6) -> tuple[RouteName, float]:
        # Embed the user query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        best_route = RouteName.UNKNOWN
        best_score = 0.0
        
        # Calculate cosine similarity with each route
        for route, route_embedding in self.route_examples:
            score = util.cos_sim(query_embedding, route_embedding).item()
            if score > best_score:
                best_score = score
                best_route = route
                
        # If the best match is below our confidence threshold, return unknown
        if best_score < threshold:
            return RouteName.UNKNOWN, best_score
            
        return best_route, best_score