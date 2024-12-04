from reliabilipy import reliability_analysis
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# LLM judge types
judge_model_types = ['Nexusflow/Starling-LM-7B-beta',
                     'meta-llama/Meta-Llama-3-8B-Instruct',
                     'google/gemma-1.1-7b-it']

resp_model_types = ['lmsys/vicuna-7b-v1.5',
                    'tiiuae/falcon-7b']


# Load judgement data by model
judgements_by_mod  = {'lmsys/vicuna-7b-v1.5':{}, 'tiiuae/falcon-7b':{}}
for jud_mod in judge_model_types:
    for resp_mod in resp_model_types:
        jud_mod_print_name = jud_mod.replace('/', '_')
        resp_mod_print_name = resp_mod.replace('/', '_')
        with open(f'judge_full_{jud_mod_print_name}_by_{resp_mod_print_name}.json','r') as resp:
            judgements_by_mod[resp_mod][jud_mod] = json.load(resp)
 

def judge_to_numeric(judged_prompts):
  '''
  Args:
    judged_prompts: list of the model's judgement replications for each prompt for each model
  Returns:
    dict of the judgements represented numerically
  '''
  num_judge_by_mod = {}
  for mod, jud_dict in judged_prompts.items():
    num_judge_by_mod[mod] = {}
    for prompt, judgements in jud_dict.items():
      curr_judgements = []
      for jud in judgements:
          # Non-response considered to be the response 'incorrect'
          if 'correct' in jud.lower() and 'incorrect' not in jud.lower():
            res = 1 # correct
          else:
            res = 0 # incorrect
          curr_judgements.append(res)
      num_judge_by_mod[mod][prompt] = curr_judgements
  return num_judge_by_mod

def judge_reliab(resp_mod, judged_prompts_num):
  '''
  Args:
    resp_mod: name of the model that is being judged
    judged_prompts_num: dict of the numerical representation for the model's judgement replications for each prompt
  Returns:
    dict of the reliability of judgements of the model
  '''
  results = {}
  df = pd.DataFrame.from_dict(judged_prompts_num)
  # Drop columns where LLM makes no judgements
  full_1_reliab_count = 0
  for col in df.columns:
    if len(df[col].unique()) == 1:
      full_1_reliab_count += 1
      df.drop(col, axis=1, inplace=True)
  if df.shape[1] == 0:
    full_reliab_omeg = 1
  else:
    full_reliab = reliability_analysis(raw_dataset=df, is_corr_matrix=False)
    full_reliab.fit()
    full_reliab_omeg = ((full_1_reliab_count)+((df.shape[1])*full_reliab.omega_total))/(full_1_reliab_count+df.shape[1])
  results = full_reliab_omeg
  return results


num_judge = {'lmsys/vicuna-7b-v1.5':{}, 'tiiuae/falcon-7b':{}}
for resp_mod, jud_dict in judgements_by_mod.items():
  num_judge[resp_mod] = judge_to_numeric(jud_dict)
  
alm_dfs = {}
for resp_mod, resp_dict in num_judge.items():
  alm_dfs[resp_mod]={}
  for mod, jud_dict in resp_dict.items():
    alm_dfs[resp_mod][mod] = pd.DataFrame(jud_dict).mean(axis=1)
  alm_dfs[resp_mod]=pd.DataFrame(alm_dfs[resp_mod])

  
jud_reliab = {}
for resp_mod, resp_dict in num_judge.items():
  jud_reliab[resp_mod]={}
  for mod, jud_dict in resp_dict.items():
    jud_reliab[resp_mod][mod] = judge_reliab(resp_mod, jud_dict)
with open(f'judge_application_reliab_omega.txt', 'w') as file:
    pd.DataFrame(jud_reliab).to_csv(file, index=False)
  
  
# A_LM Plots: uncomment if needed
#plt.figure(figsize=(10, 6))
#bp = plt.boxplot(alm_dfs['lmsys/vicuna-7b-v1.5'], labels=['Starling-LM-7B-beta', 'Meta-Llama-3-8B-Instruct', 'Gemma-1.1-7b-it'], patch_artist=True)
#
#colors = ['lightblue', 'lightgreen', 'lightcoral']
#for patch, color in zip(bp['boxes'], colors):
#    patch.set_facecolor(color)
#    patch.set_edgecolor('black')
#for median in bp['medians']:
#    median.set_color('black')
#plt.xticks(fontsize=12)
#plt.title(r'Vicuna (7B) $A_{LM}$ Comparison', fontsize=16)
#plt.xlabel('Judgment Model', fontsize=14)
#plt.ylabel(r'$A_{LM}$', fontsize=14)
#plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.savefig('vicuna_performance_comparison.png', bbox_inches='tight')
#plt.show()
  

