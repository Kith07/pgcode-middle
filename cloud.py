from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationMixin
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig
import os
from typing import Callable, List, Optional, Tuple, Union
from transformers import AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
# from .config import XCodeConfig
import torch.nn as nn
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers import set_seed
from datasets import load_dataset, concatenate_datasets, DatasetDict, load_from_disk
# from utils.prompter import Prompter
import transformers
import time
class XCodeMiddleConfig(PretrainedConfig):
    model_type = "xcodemiddle"

    def __init__(
        self,
        enc_dec_origin_model: Optional[str] = None,
        enc_num_layers: int = 1,
        middle_path: Optional[str] = None,
        dec_num_layers: int = 4,
        middle_num_layers: Optional[int] = None,
        is_middle_api: bool = False,
        enc_config:dict = {},
        middle_config:dict = {},
        dec_config:dict = {},
        other_config_path: Optional[str] = None,
        **kwargs,
    ):
        self.enc_dec_origin_model = enc_dec_origin_model
        self.enc_num_layers = enc_num_layers
        self.middle_path = middle_path
        self.is_middle_api = is_middle_api
        self.middle_num_layers = middle_num_layers
        self.dec_num_layers = dec_num_layers
        self.enc_config = enc_config
        self.middle_config = middle_config
        self.dec_config = dec_config
        if other_config_path:
            other_config_dict = AutoConfig.from_pretrained(other_config_path).to_dict()
        else:
            other_config_dict = {}
        super().__init__(**other_config_dict ,**kwargs)

class MiddleXCodeModel(PreTrainedModel):
    config_class = XCodeMiddleConfig

    def __init__(self, config: XCodeMiddleConfig):
        super().__init__(config)
        middle_config = AutoConfig.from_pretrained(config.middle_path)
        # self.middle = AutoModel.from_config(
        #         middle_config
        #     )
        self.middle = AutoModel.from_config(
                middle_config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
            )
        self.middle.layers = self.middle.layers[config.enc_num_layers: config.num_hidden_layers - config.dec_num_layers]
        #self.middle.layers = self.middle.layers[config.enc_num_layers: config.num_hidden_layers]
        self.middle.norm = nn.Identity()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = self.middle(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        #     position_ids=position_ids,
            # past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
            use_cache=False,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        )
        return output

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from typing import List, Optional, Dict
import numpy as np
from transformers import AutoConfig
import uvicorn

# Import your model classes
# from model import XCodeMiddleConfig, MiddleXCodeModel  # Assuming your model code is in model.py
AutoConfig.register("xcodemiddle", XCodeMiddleConfig)
AutoModel.register(XCodeMiddleConfig, MiddleXCodeModel)

app = FastAPI(title="XCode Model API")
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
# Pydantic models for request/response
class ModelInput(BaseModel):
    inputs_embeds: List[List[List[float]]]
    # attention_mask: Optional[List[int]]

class ModelOutput(BaseModel):
    last_hidden_state: List[List[List[float]]]
    middle_time: float
    conversion_time_1: float
    conversion_time_2: float
    concat_time: float
    # Add other fields based on your model's output structure

# Global variable to store the model
model = None
last_embedding = None
try:

        
    # Initialize the model
    model = AutoModel.from_pretrained("./qwen/mid", torch_dtype=torch.bfloat16)
    
    # Load model weights if needed
    # model.load_state_dict(torch.load("path/to/weights"))
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model")
@app.post("/clear")
async def clear():
    global last_embedding
    last_embedding = None

@app.get("/last")
async def get():
    global last_embedding
    print(last_embedding)

@app.post("/predict", response_model=ModelOutput)
async def predict(input_data: ModelInput):
    """
    Endpoint to run inference with the model
    """
    global last_embedding
    # Convert input lists to tensors
    conversion_start_1 = time.time()
    inputs_embeds = torch.tensor(input_data.inputs_embeds, dtype=torch.bfloat16, device="cuda")
    attention_mask = None
    # if input_data.attention_mask:
    #     attention_mask = torch.tensor(input_data.attention_mask)
        
    # Move inputs to GPU if model is on GPU
    if torch.cuda.is_available():
        inputs_embeds = inputs_embeds.cuda()
        
        # if attention_mask is not None:
        #     attention_mask = attention_mask.cuda()
    conversion_end_1 = time.time()
    concat_start = time.time()
    if last_embedding is None:
            inputs_embeds = inputs_embeds
    else:
        inputs_embeds = torch.cat(
            (last_embedding, inputs_embeds), dim=1
        )
    concat_end = time.time()
    middle_time_start = time.time()
    # Run inference
    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            # attention_mask=attention_mask,
        )

    middle_time_end = time.time()
    conversion_start_2 = time.time()
    last_hidden_state = None
    # Convert output tensors to lists for JSON serialization   
    if last_embedding is None:
        last_hidden_state = outputs.last_hidden_state.cpu().to(torch.float16).tolist()
    else:
        last_hidden_state = outputs.last_hidden_state[:,-1:,:].cpu().to(torch.float16).tolist()

    last_embedding = inputs_embeds
    conversion_end_2 = time.time()


    try:
        return ModelOutput(
            last_hidden_state=last_hidden_state,
            middle_time = middle_time_end - middle_time_start,
            conversion_time_1 =  (conversion_end_1 - conversion_start_1),
            conversion_time_2 = (conversion_end_2 - conversion_start_2),
            concat_time = concat_end - concat_start
        )
    except Exception as e:
        print(e)
        

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)