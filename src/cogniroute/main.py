from dotenv import load_dotenv
load_dotenv() # This loads the GEMINI_API_KEY from the .env file

from fastapi import FastAPI
from .schemas import RouterRequest, AgentResponse, RouteName
from .router import SemanticRouter
from .agents import generate_explanation, generate_quiz, generate_study_plan
from .memory import add_memory, get_relevant_memories

app = FastAPI(title="CogniRoute", description="Multi-agent semantic routing system with memory")

router = SemanticRouter()

@app.get("/")
def read_root():
    return {"status": "CogniRoute is running"}

@app.post("/process", response_model=AgentResponse)
def process_query(request: RouterRequest):
    # 1. Route the request locally
    route, confidence = router.route(request.query)
    
    # 2. If unknown, reject immediately
    if route == RouteName.UNKNOWN:
        return AgentResponse(
            route=route,
            confidence=confidence,
            data={"message": "I am a specialized educational AI. I can only explain concepts, generate quizzes, or build study plans. Please rephrase your request."}
        )
    
    # 3. Fetch user's past memories
    context = get_relevant_memories(request.query, request.user_id)
    
    # 4. Dispatch to the correct Sub-Agent, passing the context
    agent_data = None
    if route == RouteName.EXPLAIN_CONCEPT:
        agent_data = generate_explanation(request.query, context)
    elif route == RouteName.GENERATE_QUIZ:
        agent_data = generate_quiz(request.query, context)
    elif route == RouteName.STUDY_PLAN:
        agent_data = generate_study_plan(request.query, context)
        
    # 5. Save the current query to memory for future use
    add_memory(request.query, route.value, request.user_id)
        
    # 6. Return the structured data
    return AgentResponse(
        route=route,
        confidence=confidence,
        data=agent_data.model_dump() if agent_data else {}
    )