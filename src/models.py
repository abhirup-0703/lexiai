from typing import List, Dict, Optional, Annotated
import operator
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator
from typing_extensions import TypedDict

def normalize_rubric(v):
    if isinstance(v, str):
        return {"criteria": v, "key_concepts": []}
    return v

class GradingRubric(BaseModel):
    key_concepts: List[str] = Field(default=[], alias="concepts") 
    criteria: str = Field(..., alias="grading_criteria")
    model_config = ConfigDict(populate_by_name=True)

class ExamQuestion(BaseModel):
    blooms_level: str = Field(..., alias="type") 
    question: str
    context_snippet: str = Field(..., description="Verbatim quote from the text.")
    rubric: Annotated[GradingRubric, BeforeValidator(normalize_rubric)] = Field(..., alias="grading_rubric")
    model_config = ConfigDict(populate_by_name=True)

class ExamPlan(BaseModel):
    topic: str = Field(..., alias="research_context")
    questions: List[ExamQuestion]
    model_config = ConfigDict(populate_by_name=True)

# LangGraph State
class InterviewState(TypedDict):
    exam_plan: dict                 # JSON dump of ExamPlan
    current_q_index: int            
    history: Annotated[List[str], operator.add] 
    last_judge_result: Optional[dict] 
    retry_count: int