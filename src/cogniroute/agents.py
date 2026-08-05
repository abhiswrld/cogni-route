import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from .schemas import QuizResponse, ExplanationResponse, StudyPlanResponse

# Initialize the Gemini Client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def generate_explanation(query: str) -> ExplanationResponse:
    print(f"[Agent] Calling Gemini for Explanation: {query}")
    prompt = f"You are an expert educational tutor. Provide a clear explanation of the user's query. Break it down into a summary and 3 key points.\n\nUser Query: {query}"
    
    # The new SDK uses client.models.generate_content
    result = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ExplanationResponse,
        },
    )
    return ExplanationResponse.model_validate_json(result.text)

def generate_quiz(query: str) -> QuizResponse:
    print(f"[Agent] Calling Gemini for Quiz: {query}")
    prompt = f"You are a quiz generator. Create exactly 5 multiple-choice questions with 4 options each based on the user's request.\n\nUser Query: {query}"
    
    result = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": QuizResponse,
        },
    )
    return QuizResponse.model_validate_json(result.text)

def generate_study_plan(query: str) -> StudyPlanResponse:
    print(f"[Agent] Calling Gemini for Study Plan: {query}")
    prompt = f"You are a study planner. Create a concise, bulleted study timeline based on the user's request.\n\nUser Query: {query}"
    
    result = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": StudyPlanResponse,
        },
    )
    return StudyPlanResponse.model_validate_json(result.text)