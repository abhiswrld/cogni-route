from pydantic import BaseModel, Field
from enum import Enum

"""
Create a strict list of strings. The route must be one of the following, since Enum is used,
it will raise a validation error if the route is not one of the defined values.
"""
class RouteName(str, Enum):
    EXPLAIN_CONCEPT = "explain_concept"
    GENERATE_QUIZ = "generate_quiz"
    STUDY_PLAN = "study_plan"
    UNKNOWN = "unknown"

"""
This defines the shape of the data coming into the API. Query is a required field, and it must be a string.
Field(...) means that this field is required. The description is used for documentation purposes.
"""
class RouterRequest(BaseModel):
    query: str = Field(..., description="The user's natural language input")

"""
This defines the shape of the data going out of the API. It includes the original query, the matched route, and a confidence score.
"""
class RouterResponse(BaseModel):
    query: str
    route: RouteName
    confidence: float = Field(..., description="Cosine similarity score of the matched route")