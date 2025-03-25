from fastapi import APIRouter, HTTPException
from models.models import EncoderInput, MiddleInput, DecoderInput, GenerationOutput
from services.encoder_service import EncoderService
from services.middle_service import MiddleService
from services.decoder_service import DecoderService
from core.session import SessionManager
import time
import logging

router = APIRouter(tags=["generation"])
logger = logging.getLogger(__name__)

@router.post("/generate", response_model=GenerationOutput)
async def generate(input_data: EncoderInput):
    """
    Encoder -> Middle -> Decoder
    """
    session_id = input_data.session_id
    start_time = time.time()
    
    if session_id:
        await SessionManager.create_session(session_id)
    
    encoder_start = time.time()
    encoder_output = await EncoderService.encode(input_data)
    encoder_time = time.time() - encoder_start

    if isinstance(encoder_output, dict):
        if "output" in encoder_output:
            inputs_embeds = encoder_output["output"]
        elif "last_hidden_state" in encoder_output:
            inputs_embeds = encoder_output["last_hidden_state"] 
        else:
            print(f"Unexpected encoder output structure. Keys: {encoder_output.keys()}")
            inputs_embeds = encoder_output
    else:
        inputs_embeds = encoder_output

    middle_start = time.time()
    middle_input = MiddleInput(inputs_embeds=inputs_embeds)
    middle_output = await MiddleService.predict(middle_input)
    middle_time = time.time() - middle_start
    
    decoder_start = time.time()
    decoder_input = DecoderInput(
        hidden_states=middle_output["last_hidden_state"],
        prompt=input_data.prompt,
        max_new_tokens=input_data.max_new_tokens,
        temperature=input_data.temperature,
        top_p=input_data.top_p,
        session_id=session_id,
        stream=input_data.stream
    )
    
    decoder_output = await DecoderService.decode(decoder_input)
    decoder_time = time.time() - decoder_start
    
    total_time = time.time() - start_time
    
    return GenerationOutput(
        generated_text=decoder_output["generated_text"],
        processing_time=total_time,
        encoder_time=encoder_time,
        middle_time=middle_time,
        decoder_time=decoder_time
    )