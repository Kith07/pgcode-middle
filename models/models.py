from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class EncoderInput(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    session_id: Optional[str] = None
    stream: Optional[bool] = False

class EncoderOutput(BaseModel):
    hidden_states: List[List[List[float]]]

class MiddleInput(BaseModel):
    inputs_embeds: List[List[List[float]]]

class MiddleOutput(BaseModel):
    last_hidden_state: List[List[List[float]]]
    middle_time: float
    conversion_time_1: float
    conversion_time_2: float
    concat_time: float

class DecoderInput(BaseModel):
    hidden_states: List[List[List[float]]]
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    session_id: Optional[str] = None
    stream: bool = False

class DecoderOutput(BaseModel):
    generated_text: str

class GenerationOutput(BaseModel):
    generated_text: str
    processing_time: float
    encoder_time: float
    middle_time: float
    decoder_time: float

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]

class SessionResponse(BaseModel):
    status: str
    message: str
    session_id: str