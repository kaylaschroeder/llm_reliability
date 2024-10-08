!pip install reliabiliPy

import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch
import pandas as pd
from reliabilipy import reliability_analysis

import utils

device = "cuda:0"

# Hugging face access token
access_token = ''

# Define LLM judge types
model_types = ['meta-llama/Meta-Llama-3-8B',
               'mistralai/Mistral-7B-v0.1',
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

judge_prompt = """You are a fair and objective judge tasked with selecting the strongest of the following responses to the
provided question based upon the accuracy, utility, and relevance of each. Do not consider length of response, positioning
of response or title of response in your judgement. Output exclusively the letter of the best response and nothing else.
"""

  prompt_resp_formatted = {}
for prompt, resp_dict in prompt_resp.items():
  prompt_resp_formatted[prompt] = {'type': resp_dict['type'], 'model_responses': list(resp_dict['model_responses'].values())}

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
    judge_bbh = [[generate_txt(model, tokenizer, judge_prompt+"\nQuestion: "+q+"\n[A]: "+\
      resp['model_responses'][0]+\
      "\n[B]: "+resp['model_responses'][1]+ \
      "\n[C]: "+resp['model_responses'][2]+\
      "\n[D]: "+ resp['model_responses'][3]+\
      "\n[E]: "+resp['model_responses'][4],
      indiv_seed) for indiv_seed in judge_seeds]\
      for q, resp in prompts_judge.items() if resp['type'] == 'single']
    with open(f'judge_bbh_{model_name}.json','w') as f:
          json.dump(judge_bbh, f)
    judge_mt = [[generate_txt(judge_prompt+"\nQuestion: "+q[1]+"\n[A]: "+\
    resp['model_responses'][0]+\
    "\n[B]: "+resp['model_responses'][1]+ \
    "\n[C]: "+resp['model_responses'][2]+\
    "\n[D]: "+resp['model_responses'][3]+\
    "\n[E]: "+resp['model_responses'][4]
    ,indiv_seed) for indiv_seed in judge_seeds]\
    for q, resp in prompts_judge.items() if resp['type']== 'multi']
    # Save outputs to json
    with open(f'judge_mt_{model_name}.json', 'w') as f:
        json.dump(judge_mt, f)
  return [judge_bbh,judge_mt]
  
  
def judge_reliab(mod_name, judged_prompts):
  '''
  Args:
    model_name: Hugging Face model name
    prompts_judge: list of the model's judgement replications for each prompt
  Returns:
    df of the reliability of the model's judgements across replications
  '''
  letter_to_number = {'[A]': 1, '[B]': 2, '[C]': 3, '[D]': 4, '[E]': 5}
  judge_bbh_numeric = [[letter_to_number[indiv_judge] for indiv_judge in prompt_judge] for prompt_judge in judged_prompts[0]]
  judge_mt_numeric = [[letter_to_number[indiv_judge] for indiv_judge in prompt_judge] for prompt_judge in judged_prompts[1]]
  bbh_reliab = reliability_analysis(raw_dataset=pd.DataFrame(judge_bbh_numeric).T, is_corr_matrix=False)
  bbh_reliab.fit()
  mtb_reliab = reliability_analysis(raw_dataset=pd.DataFrame(judge_mt_numeric).T, is_corr_matrix=False)
  mtb_reliab.fit()
  full_reliab = reliability_analysis(raw_dataset=pd.DataFrame(judge_bbh_numeric+ judge_mt_numeric).T, is_corr_matrix=False)
  full_reliab.fit()
  results = {
        "alpha": [bbh_reliab.alpha_cronbach, mtb_reliab.alpha_cronbach, full_reliab.alpha_cronbach],
        "omega": [bbh_reliab.omega_total, mtb_reliab.omega_total, full_reliab.omega_total]
  }
  results_df = pd.DataFrame(results, index=["BBH", "MTB","Full"])
  results_df.to_csv(f"judge_reliab_{mod_name}.csv")
  return results_df
  

judgements_by_mod = {}
reliability_by_mod = {}
for mod in model_types:
    judgements_by_mod[mod] = model_judge(mod, prompt_resp_lst, access_token)
    reliability_by_mod[mod] = judge_reliab(mod, judgements_by_mod[mod])


