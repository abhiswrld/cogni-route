import sys
from contextlib import redirect_stdout
from mcp.server.fastmcp import FastMCP
from .router import SemanticRouter
from .agents import generate_explanation, generate_quiz, generate_study_plan
from .memory import add_memory, get_relevant_memories
from .schemas import RouteName

# Initialize the MCP server
mcp = FastMCP("CogniRoute")

print("Initializing CogniRoute...", file=sys.stderr)

# Redirect stdout during model loading
with redirect_stdout(sys.stderr):
    router = SemanticRouter()

@mcp.tool()
def process_educational_query(query: str, user_id: str = "claude-user") -> dict:
    """
    Routes an educational query, calls a specialized AI agent, and returns structured JSON.
    Can explain concepts, generate quizzes, or create study plans.
    Maintains memory of past interactions for the given user_id.
    
    IMPORTANT FORMATTING RULE: When returning a quiz to the user, DO NOT reveal the correct_answer. 
    Only show the question and the options. Ask the user to guess first, then offer to reveal the answers.
    """
    # Wrap the entire execution in redirect_stdout so agents.py prints don't corrupt the JSON stream
    with redirect_stdout(sys.stderr):
        print(f"[MCP] Received query from {user_id}: {query}")
        
        # 1. Route
        route, confidence = router.route(query)
        
        if route == RouteName.UNKNOWN:
            return {
                "status": "rejected",
                "message": "I am a specialized educational AI. I can only explain concepts, generate quizzes, or build study plans."
            }
        
        # 2. Memory Retrieval
        context = get_relevant_memories(query, user_id)
        
        # 3. Dispatch to Agent
        agent_data = None
        if route == RouteName.EXPLAIN_CONCEPT:
            agent_data = generate_explanation(query, context)
        elif route == RouteName.GENERATE_QUIZ:
            agent_data = generate_quiz(query, context)
        elif route == RouteName.STUDY_PLAN:
            agent_data = generate_study_plan(query, context)
            
        # 4. Save to Memory
        add_memory(query, route.value, user_id)
        
        # 5. Return the structured data back to Claude
        return {
            "status": "success",
            "route_taken": route.value,
            "router_confidence": confidence,
            "agent_response": agent_data.model_dump() if agent_data else {}
        }

if __name__ == "__main__":
    print("Starting CogniRoute MCP Server...", file=sys.stderr)
    mcp.run(transport='stdio')