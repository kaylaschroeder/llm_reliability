import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch

import utils

device = "cuda:0"

# Define LLM types
model_types = ['meta-llama/Meta-Llama-3-8B',
               'mistralai/Mistral-7B-v0.1',
               'lmsys/vicuna-7b-v1.5',
               'google/gemma-7b',
               'bigscience/bloom-7b1',
               'microsoft/phi-2',
               'tiiuae/falcon-7b']
               
# Access token for hugging face
access_token = ''
               
# Load sampled questions
with open('bbh_sampled_q.json', 'r') as bbh:
    bbh_qs = json.load(bbh)
with open('mt_bench_sampled_q.json', 'r') as mt_bench:
    mt_bench_qs = json.load(mt_bench)
num_qs = len(bbh_qs) + len(mt_bench_qs)
all_qs = bbh_qs + mt_bench_qs

# Randomly sample 5 models used for responses from benchmark questions
random.seed(1234)
model_per_qs = [random.sample(model_types, k = 5) for _ in range(num_qs)]

# Prompts to run per model
model_prompts = {}
for model in model_types:
  model_prompts[model] = [prompt for prompt, q_mods in zip(all_qs, model_per_qs) if model in q_mods]
  
chain_of_thought_addendum = 'Include step-by-step reasoning in answering the following question: \n'


def responses_model(model_name, model_prompts, access_token):
  '''
  Args:
    model_name: Hugging Face model name
    model_prompts: list of prompts for each model
    access_token: Hugging Face access token
  Returns:
    list of model responses for each prompt
  '''
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
  model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

  model_dict_list = []
  for prompt_dict in model_prompts:
    if 'input' in prompt_dict.keys():
      turns = 'single'
      response = generate_txt(model, tokenizer,
                              chain_of_thought_addendum + prompt_dict['input'],
                              seed_val=1234)
    else:
      turns = 'multi'
      response1 = generate_txt(model, tokenizer,
                               prompt_dict['turns'][0],
                               seed_val=1234)
      response2 = generate_txt(model, tokenizer,
                               prompt_dict['turns'][0]+response1+prompt_dict['turns'][1],
                               seed_val = 1234)
      response = response2
    model_dict = {'prompt': prompt_dict,
                  'response': response,
                  'type': turns}
    model_dict_list.append(model_dict)
  return responses

model_responses = {}
for model_name, prompts in model_prompts.items():
  responses = responses_model(model_name, prompts)
  model_responses[model_name] = responses
  
# Save outputs to json
with open('model_responses.json', 'w') as f:
    json.dump(model_responses, f)
  
# Realign prompt responses by prompt from the model outputs
prompt_responses = {}
for model_name, response_list in model_responses.items():
        for response_dict in response_list:
            prompt_dict = response_dict['prompt']
            prompt_key = prompt_dict['prompt']
            if prompt_key not in prompt_responses:
                prompt_responses[prompt_key] = {
                    'type': response_dict['type'],
                    'model_responses': {}
                }
            prompt_responses[prompt_key]['model_responses'][model_name] = response_dict['response']

# Save outputs to json
with open('prompt_responses.json', 'w') as f:
    json.dump(prompt_responses, f)

