from pydantic import BaseModel, Field
from enum import Enum
from typing import List

class RouteName(str, Enum):
    EXPLAIN_CONCEPT = "explain_concept"
    GENERATE_QUIZ = "generate_quiz"
    STUDY_PLAN = "study_plan"
    UNKNOWN = "unknown"

class RouterRequest(BaseModel):
    query: str = Field(..., description="The user's natural language input")

# Agent Schemas

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: str

class QuizResponse(BaseModel):
    topic: str
    questions: List[QuizQuestion]

class ExplanationResponse(BaseModel):
    concept: str
    summary: str
    key_points: List[str]

class StudyPlanResponse(BaseModel):
    goal: str
    timeline: List[str]

# The main response can be any of these
class AgentResponse(BaseModel):
    route: RouteName
    confidence: float
    data: dict