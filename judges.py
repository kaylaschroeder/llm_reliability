import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch

from utils import generate_judge


device = "cuda:0"

# Hugging face access token
access_token = ''

# Define LLM judge types
model_types = ['meta-llama/Meta-Llama-3-8B',
#               'mistralai/Mistral-7B-v0.1',
               'lmsys/vicuna-7b-v1.5',
               'google/gemma-7b',
               'bigscience/bloom-7b1',
               'microsoft/phi-2',
               'tiiuae/falcon-7b']
               

# Load data for judgement
with open('prompt_responses.json', 'r') as resp:
    prompt_resp = json.load(resp)
    
# Seeds for LLM as judge
random.seed(2)
judge_seeds = random.sample(range(1,10000),100)

judge_prompt = """You are a fair and objective judge tasked with selecting the strongest of the following responses to the provided question. Base your judgement upon the accuracy, utility, and relevance of each. Do not consider length of response, positioning of response or title of response in your judgement. Output exclusively the letter of the best response and nothing else.
"""

judge_prompt_multi = """You are a fair and objective judge tasked with selecting the strongest of the following responses to the second provided question ("Question 2"). Question 1 is only provided for context. Base your judgement upon the accuracy, utility, and relevance of each. Do not consider length of response, positioning of response or title of response in your judgement. Output exclusively the letter of the best response and nothing else.
"""

prompt_resp_formatted = {}
for prompt_label, resp_dict in prompt_resp.items():
  prompt_resp_formatted[prompt_label] = {'type': resp_dict['type'],
                                   'prompt': resp_dict['prompt'],
                                   'model_responses': list(resp_dict['model_responses'].values())}

def model_judge(model_name, prompts_judge, access_token):
  '''
  Args:
    model_name: Hugging Face model name
    prompts_judge: dictionary of prompts to be judged by the model
    access_token: Hugging Face access token
  Returns:
    list of the model's judgement replications for each prompt
  '''
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
  model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

  model_judge_list = []
  for prompt_dict in prompts_judge:
    judge_bbh = [[generate_judge(model, tokenizer, judge_prompt+"\nQuestion: "+\
      q_dict['prompt']+"Responses: \n[A]: "+\
      q_dict['model_responses'][0]+\
      "\n[B]: "+ q_dict['model_responses'][1]+ \
      "\n[C]: "+ q_dict['model_responses'][2]+\
      "\n[D]: "+ q_dict['model_responses'][3]+\
      "\n[E]: "+ q_dict['model_responses'][4],
      indiv_seed, device) for indiv_seed in judge_seeds]\
      for q, q_dict in prompts_judge.items() if q_dict['type'] == 'single']
    mod_print_name = model_name.replace('/', '_')
    with open(f'judge_bbh_{mod_print_name}.json','w') as f:
          json.dump(judge_bbh, f)
    judge_mt = [[generate_judge(judge_prompt_multi+"\nQuestion 1: "+q_dict['prompt'][0]+\
      "\nQuestion 2: "+q_dict['prompt'][1]+\
      "Responses: \n[A]: "+ q_dict['model_responses'][0]+\
      "\n[B]: "+q_dict['model_responses'][1]+ \
      "\n[C]: "+q_dict['model_responses'][2]+\
      "\n[D]: "+q_dict['model_responses'][3]+\
      "\n[E]: "+q_dict['model_responses'][4],
      indiv_seed, device) for indiv_seed in judge_seeds]\
    for q, q_dict in prompts_judge.items() if q_dict['type']== 'multi']
    # Save outputs to json
    with open(f'judge_mt_{mod_print_name}.json', 'w') as f:
        json.dump(judge_mt, f)
  return [judge_bbh,judge_mt]


judgements_by_mod = {}
for mod in model_types:
    judgements_by_mod[mod] = model_judge(mod, prompt_resp_formatted, access_token)

# Save judgements outputs to json
with open('full_judgements.json', 'w') as f:
    json.dump(judgements_by_mod, f)


