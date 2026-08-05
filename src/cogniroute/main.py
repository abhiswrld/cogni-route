from dotenv import load_dotenv
load_dotenv() # This reads the .env file and sets environment variables, including OPENAI_API_KEY

from fastapi import FastAPI
from .schemas import RouterRequest, AgentResponse, RouteName
from .router import SemanticRouter
from .agents import generate_explanation, generate_quiz, generate_study_plan

app = FastAPI(title="CogniRoute", description="Multi-agent semantic routing system")

router = SemanticRouter()

@app.get("/")
def read_root():
    return {"status": "CogniRoute is running"}

@app.post("/process", response_model=AgentResponse)
def process_query(request: RouterRequest):
    # 1. Route the request locally
    route, confidence = router.route(request.query)
    
    # 2. If we don't know what they want, reject it immediately. No OpenAI call is made.
    if route == RouteName.UNKNOWN:
        return AgentResponse(
            route=route,
            confidence=confidence,
            data={"message": "I am a specialized educational AI. I can only explain concepts, generate quizzes, or build study plans. Please rephrase your request."}
        )
    
    # 3. Dispatch to the correct Sub-Agent (makes the OpenAI call)
    agent_data = None
    if route == RouteName.EXPLAIN_CONCEPT:
        agent_data = generate_explanation(request.query)
    elif route == RouteName.GENERATE_QUIZ:
        agent_data = generate_quiz(request.query)
    elif route == RouteName.STUDY_PLAN:
        agent_data = generate_study_plan(request.query)
        
    # 4. Return the structured data
    return AgentResponse(
        route=route,
        confidence=confidence,
        data=agent_data.model_dump() if agent_data else {}
    )