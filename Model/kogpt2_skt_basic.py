#!/usr/bin/env python
# coding: utf-8

# In[16]:


import gc
import wandb
import pandas as pd
import torch
import torch.cuda


# In[17]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.getcwd()


# In[18]:


from pathlib import Path
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from transformers import TextDataset,DataCollatorForLanguageModeling


# In[19]:


get_ipython().system('pip install wandb')
get_ipython().system('pip install ipywidgets')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install whatlies[all]')


# In[20]:


torch.__version__
torch.cuda.is_available()
get_ipython().system('nvidia-smi')


# In[21]:


def read_txt(file_path):
    with open(file_path, 'r', encoding ="UTF-8") as f:
    #with open(file_path, 'r') as f:
        fr = f.read()
    return fr

def read_csv(file_path):
    fr = pd.read_csv(file_path, encoding = "utf-8")
    return fr


# In[22]:


dataset = read_csv('/home/danbi/userdata/SGM_AI/darklady/dataset/dataset_0831.csv')
dataset


# In[23]:


swapped_dataset = read_csv('/home/danbi/userdata/SGM_AI/darklady/dataset/swapped_data_0822.csv')
swapped_dataset#['동화']


# In[24]:


concat_dataset = dataset + swapped_dataset
concat_dataset['동화'][0]


# In[25]:


trainset, testset = train_test_split(concat_dataset['동화'], test_size=0.2, shuffle=True, random_state=123)
#trainset, testset

fw = open("/home/danbi/userdata/SGM_AI/darklady/dataset/tmp_trainset.txt", "w", encoding = "utf-8")
fw.write(trainset.str.cat(sep=' ').replace('\n', ' '))
fw.close()

fw = open("/home/danbi/userdata/SGM_AI/darklady/dataset/tmp_testset.txt", "w", encoding = "utf-8")
fw.write(testset.str.cat(sep=' ').replace('\n', ' '))
fw.close()


# In[26]:


# from transformers import PreTrainedTokenizerFast
# from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM 
import tqdm as notebook_tqdm

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2", use_cache = False,
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')


# In[27]:


from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', use_cache = False).to(device='cuda', non_blocking=True)
torch.cuda.empty_cache()


# In[32]:


prompt = """
지금부터 이야기를 시작하겠습니다. 옛날옛적에 한 오누이가 살고 있었습니다.
"""
with torch.no_grad():
  tokens_bfr = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
  gen_tokens_bfr = model.generate(tokens_bfr, do_sample=True, temperature=0.9, max_length=200, 
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              bos_token_id=tokenizer.bos_token_id)
  generated_bfr = tokenizer.batch_decode(gen_tokens_bfr)[0].replace("\n", " ")
  
print(generated_bfr)


# In[ ]:


def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_path = "/home/danbi/userdata/SGM_AI/darklady/dataset/tmp_trainset.txt"
test_path = "/home/danbi/userdata/SGM_AI/darklady/dataset/tmp_testset.txt"
train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)


# In[39]:


learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[123]:


from transformers import Trainer, TrainingArguments,AutoModelWithLMHead

training_args = TrainingArguments(
    output_dir="/home/danbi/userdata/SGM_AI/darklady/kogpt2_finetuned_0822", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=60, # number of training epochs
    per_device_train_batch_size=64, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    logging_steps=20,
    eval_steps = 100, # Number of update steps between two evaluations.
    save_steps= 15000, # after # steps model is saved 
    warmup_steps=400,# number of warmup steps for learning rate scheduler
    save_strategy = "steps",
    evaluation_strategy = "steps",
    prediction_loss_only=True,
    optim="adamw_torch",
    dataloader_pin_memory=False,
    report_to="none",
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# In[124]:


gc.collect()
torch.cuda.empty_cache()


# In[ ]:


#optimizer.zero_grad()
#with torch.no_grad():
torch.cuda.empty_cache()
trainer.train()


# In[113]:


trainer.save_model()


# In[114]:


finetuned_model_path = "--"
finetuned_model = GPT2LMHeadModel.from_pretrained(finetuned_model_path).to(device='cuda', non_blocking=True)


# In[121]:


#existing code
prompt = """
옛날 옛적에 한 오누이가 살고 있었어요.
"""
with torch.no_grad():
  tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
  gen_tokens = finetuned_model.generate(tokens, 
                                        early_stopping=True,
                                        do_sample=True,
                                        temperature=0.8, 
                                        max_length=250, 
                                        repetition_penalty=3.0,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        bos_token_id=tokenizer.bos_token_id,
                                        unk_token_id=tokenizer.unk_token_id,)
  generated = tokenizer.decode(gen_tokens[0], skip_special_tokens=True).replace("\n", " ")
  
print(generated)

