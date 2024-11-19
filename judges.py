import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch
import sys

from utils import generate_judge


device = "cuda:1"

# Hugging face access token
access_token = ''

# Define LLM judge types
model_types = ['Nexusflow/Starling-LM-7B-beta', 'meta-llama/Meta-Llama-3-8B-Instruct', 'google/gemma-1.1-7b-it']
               

# Load data for judgement
with open('prompt_responses.json', 'r') as resp:
    prompt_resp = json.load(resp)
    
# Seeds for LLM as judge
random.seed(2)
judge_seeds = random.sample(range(1,10000),100)

judge_prompt = """You are a fair and objective judge tasked with selecting the strongest of the following responses to the provided question. Base your judgement upon the accuracy, utility, and relevance of each. Do not consider length of response, positioning of response or title of response in your judgement. Output the letter of the best response followed by an explanation, and strictly follow the following format: "Best Response: [[letter]]".
"""

judge_prompt_multi = """You are a fair and objective judge tasked with selecting the strongest of the following responses to the second provided question ("Question 2"). Question 1 is only provided for context. Base your judgement upon the accuracy, utility, and relevance of each. Do not consider length of response, positioning of response or title of response in your judgement. Output the letter of the best response followed by an explanation, and strictly follow the following format: "Best Response: [[letter]]".
"""

prompt_resp_formatted = {}
for prompt_label, resp_dict in prompt_resp.items():
  prompt_resp_formatted[prompt_label] = {'type': resp_dict['type'],
                                   'prompt': resp_dict['prompt'],
                                   'model_responses': list(resp_dict['model_responses'].values())}

def model_judge(model_name, prompts_judge, temperatures, access_token):
  '''
  Args:
    model_name: Hugging Face model name
    prompts_judge: dictionary of prompts to be judged by the model
    temperatures: list of temperature parameter values
    access_token: Hugging Face access token
  Returns:
    list of the model's judgement replications for each prompt
  '''
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
  model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

  model_judge_dict = {}
  mod_print_name = model_name.replace('/', '_')
  for temp in temperatures:
    print(f'temperature: {temp}')
    temp_str = 'temperature: '+str(temp)
    model_judge_dict[temp_str] = {}
    for q, q_dict in prompts_judge.items():
        if q_dict['type'] == 'bbh':
            mod_judge = [generate_judge(model, tokenizer, judge_prompt+"\nQuestion: "+ \
            q_dict['prompt']+"Responses: \n[A]: "+ q_dict['model_responses'][0]+\
            "\n[B]: "+ q_dict['model_responses'][1]+ \
            "\n[C]: "+ q_dict['model_responses'][2]+ \
            "\n[D]: "+ q_dict['model_responses'][3]+ \
            "\n[E]: "+ q_dict['model_responses'][4],
            temp, device) for _ in range(100)]
            model_judge_dict[temp_str][q] = mod_judge
        elif q_dict['type'] == 'squad':
            mod_judge = [generate_judge(model, tokenizer, judge_prompt+ \
            q_dict['prompt']+"Responses: \n[A]: "+ q_dict['model_responses'][0]+\
            "\n[B]: "+ q_dict['model_responses'][1]+ \
            "\n[C]: "+ q_dict['model_responses'][2]+ \
            "\n[D]: "+ q_dict['model_responses'][3]+ \
            "\n[E]: "+ q_dict['model_responses'][4],
            temp, device) for _ in range(100)]
            model_judge_dict[temp_str][q] = mod_judge
        else:
            mod_judge = [generate_judge(model, tokenizer, judge_prompt_multi+"\nQuestion 1: "+\
            q_dict['prompt'][0]+"\nQuestion 2: "+q_dict['prompt'][1]+\
            "Responses: \n[A]: "+ q_dict['model_responses'][0]+\
            "\n[B]: "+q_dict['model_responses'][1]+ \
            "\n[C]: "+q_dict['model_responses'][2]+\
            "\n[D]: "+q_dict['model_responses'][3]+\
            "\n[E]: "+q_dict['model_responses'][4],
            temp, device) for _ in range(100)]
            model_judge_dict[temp_str][q] = mod_judge
    file_temp_str = str(temp)
    with open(f'judge_temp_{file_temp_str}_full_{mod_print_name}.json', 'w') as f:
        json.dump(model_judge_dict, f)
  return model_judge_dict


epsilon = sys.float_info.epsilon
temperature_values = [1, 0.75, 0.5, 0.25, epsilon]

judgements_by_mod = {}
for mod in model_types:
    judgements_by_mod[mod] = model_judge(mod, prompt_resp_formatted, temperature_values, access_token)

# Save judgements outputs to json
with open(f'full_judgements.json', 'w') as f:
    json.dump(judgements_by_mod, f)


