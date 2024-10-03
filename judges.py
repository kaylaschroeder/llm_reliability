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

prompt_resp_lst = {}
for prompt, resp_dict in prompt_resp.items():
  prompt_resp_lst[prompt] = {'type': resp_dict['type'],
                              'model_responses': list(resp_dict['model_responses'].values())}

judge_bbh = [[generate_txt(judge_prompt+"\nQuestion: "+q+"\n[A]: "+resp['model_responses'][0]+\
                           "\n[B]: "+resp['model_responses'][1]+ \
                           "\n[C]: "+resp['model_responses'][2]+\
                           "\n[D]: "+ resp['model_responses'][3]+\
                           "\n[E]: "+resp['model_responses'][4],
                           indiv_seed) for indiv_seed in judge_seeds]\
             for q, resp in prompt_resp_lst.items() if resp['type'] == 'single']
# Save outputs to json
with open('judge_bbh.json', 'w') as f:
    json.dump(judge_bbh, f)
judge_mt = [[generate_txt(judge_prompt+"\nQuestion: "+q[1]+"\n[A]: "+resp['model_responses'][0]+\
                          "\n[B]: "+resp['model_responses'][1]+ \
                           "\n[C]: "+resp['model_responses'][2]+\
                          "\n[D]: "+resp['model_responses'][3]+\
                          "\n[E]: "+resp['model_responses'][4]
                          ,indiv_seed) for indiv_seed in judge_seeds]\
             for q, resp in prompt_resp_lst.items() if resp['type']== 'multi']
# Save outputs to json
with open('judge_mt.json', 'w') as f:
    json.dump(judge_mt, f)

letter_to_number = {'[A]': 1, '[B]': 2, '[C]': 3, '[D]': 4, '[E]': 5}
judge_bbh_numeric = [[letter_to_number[indiv_judge] for indiv_judge in prompt_judge] for prompt_judge in judge_bbh]
judge_mt_numeric = [[letter_to_number[indiv_judge] for indiv_judge in prompt_judge] for prompt_judge in judge_mt]
bbh_reliab = reliability_analysis(raw_dataset=pd.DataFrame(judge_bbh_numeric).T, is_corr_matrix=False)
mtb_reliab = reliability_analysis(raw_dataset=pd.DataFrame(judge_mt_numeric).T, is_corr_matrix=False)
bbh_reliab.fit()
mtb_reliab.fit()
alpha_bbh = bbh_reliab.alpha_cronbach
alpha_mtb = mtb_reliab.alpha_cronbach
omega_bbh = bbh_reliab.omega_total
omega_mtb = mtb_reliab.omega_total

# Save results to csv
results = {
    "alpha": [alpha_bbh, alpha_mtb],
    "omega": [omega_bbh, omega_mtb]
}
results_df = pd.DataFrame(results, index=["BBH", "MTB"])
results_df.to_csv("judge_reliab.csv")
