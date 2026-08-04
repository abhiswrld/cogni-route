from fastapi import FastAPI
from .schemas import RouterRequest, RouterResponse, RouteName
from .router import SemanticRouter

app = FastAPI(title="CogniRoute", description="Multi-agent semantic routing system")

# Initialize the router on startup
router = SemanticRouter()

@app.get("/")
def read_root():
    return {"status": "CogniRoute is running"}

@app.post("/route", response_model=RouterResponse)
def route_query(request: RouterRequest):
    route, confidence = router.route(request.query)
    return RouterResponse(
        query=request.query,
        route=route,
        confidence=round(confidence, 4)
    )