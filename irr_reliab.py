import json
import re
import pandas as pd

model_types = ['Nexusflow/Starling-LM-7B-beta',
               'google/gemma-1.1-7b-it',
               'meta-llama/Meta-Llama-3-8B-Instruct']
               
judgements_by_mod  = {}
for mod in model_types:
    mod_print_name = mod.replace('/', '_')
    with open(f'full_judgements_{mod_print_name}.json','r') as resp:
        judgements_by_mod[mod] = json.load(resp)
        
def judge_to_numeric(judged_prompts):
  '''
  Args:
    judged_prompts: list of the model's judgement replications for each prompt for each model for a given temperature
  Returns:
    dict of the judgements represented numerically for a given temperature
  '''
  letter_to_number = {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0, 'E': 5.0}
  num_judge_by_mod = {}
  for mod, jud_dict in judged_prompts.items():
    num_judge_by_mod[mod] = {}
    for prompt, judgements in jud_dict.items():
      curr_judgements = []
      for jud in judgements:
          # Handles duplicate responses which result if no changes
          # Multiple valid responses provided considered to be invalid response
          res = 7.0
          jud_res = re.findall(r"Best Response:\W*([A-Ea-e])",jud)
          jud_set = set(jud_res)
          # Only 1 response provided
          if len(jud_res) == 1 or len(jud_set) == 1:
            res = letter_to_number[jud_res[0].upper()]
          # No responses provided
          elif len(jud_res) == 0:
            # No response was provided
            res = 6.0
          curr_judgements.append(res)
      num_judge_by_mod[mod][prompt] = curr_judgements
  return num_judge_by_mod
  
num_judge = {}
for mod, jud_dict in judgements_by_mod.items():
  num_judge[mod] = {}
  for temp, prompt_dict in jud_dict.items():
    num_judge[temp] = judge_to_numeric(prompt_dict)
    
irr_by_rep = []
for rep in range(100):
  irr_matches = []
  for q_type in num_judge['Nexusflow/Starling-LM-7B-beta']['temperature: 0.25'].keys():
    if q_type[0:3]=='bbh':
      curr_matches = 0
      # if num_judge['Nexusflow/Starling-LM-7B-beta']['temperature: 1'][q_type][rep] == num_judge['google/gemma-1.1-7b-it']['temperature: 1'][q_type][rep]\
      # and num_judge['Nexusflow/Starling-LM-7B-beta']['temperature: 1'][q_type][rep]!=6.0:
      if num_judge['Nexusflow/Starling-LM-7B-beta']['temperature: 0.25'][q_type][rep] == num_judge['google/gemma-1.1-7b-it']['temperature: 0.25'][q_type][rep]:
        curr_matches += 1
      # if num_judge['Nexusflow/Starling-LM-7B-beta']['temperature: 1'][q_type][rep] == num_judge['meta-llama/Meta-Llama-3-8B-Instruct']['temperature: 1'][q_type][rep]\
      # and num_judge['Nexusflow/Starling-LM-7B-beta']['temperature: 1'][q_type][rep]!=6.0:
      if num_judge['Nexusflow/Starling-LM-7B-beta']['temperature: 0.25'][q_type][rep] == num_judge['meta-llama/Meta-Llama-3-8B-Instruct']['temperature: 0.25'][q_type][rep]:
        curr_matches += 1
      # if num_judge['google/gemma-1.1-7b-it']['temperature: 1'][q_type][rep] == num_judge['meta-llama/Meta-Llama-3-8B-Instruct']['temperature: 1'][q_type][rep]\
      # and num_judge['google/gemma-1.1-7b-it']['temperature: 1'][q_type][rep]!=6.0:
      if num_judge['google/gemma-1.1-7b-it']['temperature: 0.25'][q_type][rep] == num_judge['meta-llama/Meta-Llama-3-8B-Instruct']['temperature: 0.25'][q_type][rep]:
        curr_matches += 1
      irr_matches.append(curr_matches/3)
    irr_by_rep.append(sum(irr_matches) / len(irr_matches))

from numpy import percentile

quartiles = percentile(irr_by_rep, [25, 50, 75])
# calculate min/max
data_min, data_max = min(irr_by_rep), max(irr_by_rep)
# print 5-number summary
print('Min: %.3f' % data_min)
print('Q1: %.3f' % quartiles[0])
print('Median: %.3f' % quartiles[1])
print('Q3: %.3f' % quartiles[2])
print('Max: %.3f' % data_max)

