import os
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)
import json
import torch

print(torch.__version__)
#my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
#print(my_tensor)
print(torch.cuda.is_available())


#model = torch.load('pytorch_model.bin',map_location=torch.device('cpu'))
model_path = "model_sample"
tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
config = GPT2Config.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path,config=config).to(device='cuda', non_blocking=True)
#model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')




st.title("Franklin")
prompt_text_pre = ''
prompt_text = st.text_input('Give me Prompt')
prompt_text = prompt_text_pre + prompt_text
def predict(prompt):
    # row = np.array([prompt])
    # columns = ['prompt']
    # X = pd.DataFrame([row],columns=columns)
    prompt2 = prompt
    if(len(prompt)>0):
        with torch.no_grad():
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
            gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=300).to(device='cuda', non_blocking=True)
            generated = tokenizer.decode(gen_tokens[0])
            #prediction = generated

        st.text(generated)
    else :
        st.text('Give me prompt')
    #console.log(str(generated))

st.button('Predict',on_click=predict(prompt_text))