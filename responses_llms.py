import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch

from utils import generate_txt

device = "cuda:1"

# Define LLM types
model_types = ['meta-llama/Meta-Llama-3-8B',
               'lmsys/vicuna-7b-v1.5',
               'google/gemma-7b',
               'microsoft/phi-2',
               'tiiuae/falcon-7b']
               
# Access token for hugging face
access_token = ''
               
# Load sampled questions
with open('bbh_sampled_q.json', 'r') as bbh:
    bbh_qs = json.load(bbh)
with open('mt_bench_sampled_q.json', 'r') as mt_bench:
    mt_bench_qs = json.load(mt_bench)
with open('squad_sampled_q.json', 'r') as squad:
    squad_qs = json.load(squad)
num_qs = len(bbh_qs) + len(mt_bench_qs) + len(squad_qs)
all_qs = bbh_qs + mt_bench_qs + squad_qs

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
  print(f'Loading model tokenizer for {model_name}.')
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
  print(f'Tokenizer loaded. Loading model {model_name}.')
  model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
  print(f'Model {model_name} loaded.')

  model_dict_list = []
  for prompt_dict in model_prompts:
    if 'input' in prompt_dict.keys():
      benchmark = 'bbh'
      response = generate_txt(model, tokenizer,
                              chain_of_thought_addendum + prompt_dict['input'],
                              seed_val=1234,
                              device = device)
    elif 'context' in prompt_dict.keys():
      benchmark = 'squad'
      response = generate_txt(model, tokenizer,
                              chain_of_thought_addendum +\
                              'Context: '+prompt_dict['context']+\
                              '\n Question: {'+prompt_dict['question']+\
                              ' Answer:',
                              seed_val=1234,
                              device = device)
    else:
      benchmark = 'mtb'
      response1 = generate_txt(model, tokenizer,
                               prompt_dict['turns'][0],
                               seed_val=1234,
                               device = device)
      response2 = generate_txt(model, tokenizer,
                               prompt_dict['turns'][0]+response1+prompt_dict['turns'][1],
                               seed_val = 1234,
                               device = device)
      response = response2
    model_dict = {'prompt': prompt_dict,
                  'response': response,
                  'benchmark': benchmark}
    model_dict_list.append(model_dict)
  return model_dict_list

model_responses = {}
for model_name, prompts in model_prompts.items():
  responses = responses_model(model_name, prompts, access_token)
  model_responses[model_name] = responses
  
# Save outputs to json
with open('model_responses.json', 'w') as f:
    json.dump(model_responses, f)
  
# Realign prompt responses by prompt from the model outputs
prompt_responses = {}
for model_name, response_list in model_responses.items():
  for response_dict in response_list:
    if response_dict['benchmark'] == 'mtb':
      prompt_key = response_dict['benchmark'] + ':' + response_dict['prompt']['category']
      prompts = response_dict['prompt']['turns']
    elif response_dict['benchmark'] == 'bbh':
      prompt_key = response_dict['benchmark'] + ':' + response_dict['prompt']['category']
      prompts = response_dict['prompt']['input']
    else:
      prompt_key = response_dict['benchmark'] + ':' + response_dict['prompt']['id']
      prompts = 'Context: '+ response_dict['prompt']['context'] + '\n Question: '+ response_dict['prompt']['question'] + '\n Answer:'
    if prompt_key not in prompt_responses:
      prompt_responses[prompt_key] = {
          'type': response_dict['benchmark'],
          'prompt': prompts,
          'model_responses': {}
      }
    prompt_responses[prompt_key]['model_responses'][model_name] = response_dict['response']

# Save outputs to json
with open('prompt_responses.json', 'w') as f:
    json.dump(prompt_responses, f)

