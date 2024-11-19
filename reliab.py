from reliabilipy import reliability_analysis
import pandas as pd
import json
import re

# LLM judge types
model_types = ['Nexusflow/Starling-LM-7B-beta',
               'meta-llama/Meta-Llama-3-8B-Instruct',
               'google/gemma-1.1-7b-it']

# Load judgement data by model
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

def judge_reliab(mod_name, judged_prompts_num):
  '''
  Args:
    mod_name: name of the judge model
    prompts_judge_num: list of the numerical representation for the model's judgement replications for each prompt
  Returns:
    df of the reliability of the model's judgements across replications for BBH only, SQuAD only, MT-bench only, and full set of responses
  '''
  results = {}
  for temp, temp_dict in judged_prompts_num.items():
    df = pd.DataFrame.from_dict(temp_dict)
    # Drop columns where LLM makes no judgements
    bbh_1_reliab_count = 0
    squad_1_reliab_count = 0
    mt_1_reliab_count = 0
    for col in df.columns:
      if len(df[col].unique()) == 1 and df[col].unique() == [6.0]:
        df.drop(col, axis=1, inplace=True)
      elif len(df[col].unique()) == 1:
        if col.startswith('bbh'):
          bbh_1_reliab_count += 1
        elif col.startswith('squad'):
          squad_1_reliab_count += 1
        elif col.startswith('mtb'):
          mt_1_reliab_count += 1
        df.drop(col, axis=1, inplace=True)
    full_1_reliab_count = bbh_1_reliab_count + squad_1_reliab_count + mt_1_reliab_count
    if df.shape[1] == 0:
      full_reliab_alph, full_reliab_omeg = 1, 1
    else:
      full_reliab = reliability_analysis(raw_dataset=df, is_corr_matrix=False)
      full_reliab.fit()
      full_reliab_alph = ((full_1_reliab_count)+((df.shape[1])*full_reliab.alpha_cronbach))/(full_1_reliab_count+df.shape[1])
      full_reliab_omeg = ((full_1_reliab_count)+((df.shape[1])*full_reliab.omega_total))/(full_1_reliab_count+df.shape[1])
    bbh_df = df.loc[:, df.columns.str.startswith('bbh')]
    if bbh_df.shape[1] == 0:
      bbh_reliab_alph, bbh_reliab_omeg = 1, 1
    else:
      bbh_reliab = reliability_analysis(raw_dataset=bbh_df, is_corr_matrix=False)
      bbh_reliab.fit()
      bbh_reliab_alph = ((bbh_1_reliab_count)+((bbh_df.shape[1])*bbh_reliab.alpha_cronbach))/(bbh_1_reliab_count+bbh_df.shape[1])
      bbh_reliab_omeg = ((bbh_1_reliab_count)+((bbh_df.shape[1])*bbh_reliab.omega_total))/(bbh_1_reliab_count+bbh_df.shape[1])
    squad_df = df.loc[:, df.columns.str.startswith('squad')]
    if squad_df.shape[1] == 0:
      squad_reliab_alph, squad_reliab_omeg = 1, 1
    else:
      squad_reliab = reliability_analysis(raw_dataset=squad_df, is_corr_matrix=False)
      squad_reliab.fit()
      squad_reliab_alph = ((squad_1_reliab_count)+((squad_df.shape[1])*squad_reliab.alpha_cronbach))/(squad_1_reliab_count+squad_df.shape[1])
      squad_reliab_omeg = ((squad_1_reliab_count)+((squad_df.shape[1])*squad_reliab.omega_total))/(squad_1_reliab_count+squad_df.shape[1])
    mt_df = df.loc[:, df.columns.str.startswith('mtb')]
    if mt_df.shape[1] == 0:
      mt_reliab_alph, mt_reliab_omeg = 1, 1
    else:
      mt_reliab = reliability_analysis(raw_dataset=mt_df, is_corr_matrix=False)
      mt_reliab.fit()
      mt_reliab_alph = ((mt_1_reliab_count)+((mt_df.shape[1])*mt_reliab.alpha_cronbach))/(mt_1_reliab_count+mt_df.shape[1])
      mt_reliab_omeg = ((mt_1_reliab_count)+((mt_df.shape[1])*mt_reliab.omega_total))/(mt_1_reliab_count+mt_df.shape[1])
    results[temp] = {
        'alpha': [bbh_reliab_alph, squad_reliab_alph, mt_reliab_alph, full_reliab_alph],
        'omega': [bbh_reliab_omeg, squad_reliab_omeg, mt_reliab_omeg, full_reliab_omeg]}
  alpha_data = []
  omega_data = []
  for temp, values in results.items():
    alpha_data.append([temp] + values['alpha'])
    omega_data.append([temp] + values['omega'])

  col_names = ['Temperature', 'BBH', 'SQuAD', 'MTBench', 'Full']
  df_alpha = pd.DataFrame(alpha_data, columns=col_names)
  df_omega = pd.DataFrame(omega_data, columns=col_names)

  return df_alpha, df_omega


num_judge = {}
for mod, jud_dict in judgements_by_mod.items():
  num_judge[mod] = {}
  for temp, prompt_dict in jud_dict.items():
    num_judge[temp] = judge_to_numeric(prompt_dict)
  
judge_reliab_by_mod_alpha = {}
judge_reliab_by_mod_omega = {}
for mod, mod_dict in num_judge.items():
  curr_reliab = judge_reliab(mod, mod_dict)
  judge_reliab_by_mod_alpha[mod] = curr_reliab[0]
  judge_reliab_by_mod_omega[mod] = curr_reliab[1]
  mod_print_name = mod.replace('/', '_')
  title = f'{mod_print_name} Cronbach Alpha Reliability'
  with open(f'judge_reliab_{mod_print_name}_alpha.txt', 'w') as file:
    file.write(title + '\n')
    judge_reliab_by_mod_alpha[mod].to_csv(file, index=False)
  title = f'{mod_print_name} McDonald Omega Reliability'
  with open(f'judge_reliab_{mod_print_name}_omega.txt', 'w') as file:
    file.write(title + '\n')
    judge_reliab_by_mod_omega[mod].to_csv(file, index=False)
  
