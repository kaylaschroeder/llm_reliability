import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch

from utils import generate_txt

device = "cuda:1"

# Define LLM types
model_types = ['lmsys/vicuna-7b-v1.5',
               'tiiuae/falcon-7b']
               
# Access token for hugging face
access_token = ''
               
# Load sampled questions
with open('head_to_tail_dblp.json', 'r') as dblp:
    dblp_qs = json.load(dblp)
with open('head_to_tail_mag.json', 'r') as mag:
    mag_qs = json.load(mag)

few_shot_prefix = "Answer the following questions in as few words as possible. Say 'unsure' if you don’t know. \n Question: What is the capital of China? \n  Answer: Beijing \n Question: What is the captical of Wernythedia? \n Answer: unsure \n Question: "
few_shot_suffix = " \n Answer:"

def model_answers(model_name, dblp, mag, access_token):
  ''' 
  Args:
    model_name: Hugging Face model name
    dblp: list of head to tail questions from DBLP
    mag: list of head to tail questions from MAG
    access_token: Hugging Face access token
  Returns:
    list of model responses for each prompt
  '''
  print(f'Loading model tokenizer for {model_name}.')
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
  print(f'Tokenizer loaded. Loading model {model_name}.')
  model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
  print(f'Model {model_name} loaded.')
  mod_print_name = model_name.replace('/', '_')
  resp_dict = {'mag': [], 'dblp': []}
  for q in mag['head']:
    q_resp = generate_txt(model, tokenizer,
                          few_shot_prefix+q[2]+few_shot_suffix,
                          seed_val=1234,
                          device = device)
    curr_dict = {'question': q[2], 'response': q_resp, 'ground_truth': q[3]}
    resp_dict['mag'].append(curr_dict)
  for q in dblp['head']:
    q_resp = generate_txt(model, tokenizer,
                          few_shot_prefix+q[2]+few_shot_suffix,
                          seed_val=1234,
                          device = device)
    curr_dict = {'question': q[2], 'response': q_resp, 'ground_truth': q[3]}
    resp_dict['dblp'].append(curr_dict)
  with open(f'{mod_print_name}_application_responses.json', 'w') as f:
    json.dump(resp_dict, f)
  return resp_dict


# Obtain prompts for each model
model_prompts = {}
for model in model_types:
  model_prompts[model] = model_answers(model, dblp_qs, mag_qs, access_token)

# Save outputs to json
with open('application_responses.json', 'w') as f:
    json.dump(model_prompts, f)

