from reliabilipy import reliability_analysis
import pandas as pd
import json
import re

# LLM judge types
model_types = ['Nexusflow/Starling-LM-7B-beta']

# Load judgement data by model
judgements_by_mod  = {}
for mod in model_types:
    mod_print_name = mod.replace('/', '_')
    with open(f'judge_full_{mod_print_name}.json','r') as resp:
        judgements_by_mod[mod] = json.load(resp)
 
def judge_to_numeric(judged_prompts):
  '''
  Args:
    judged_prompts: list of the model's judgement replications for each prompt for each model
  Returns:
    dict of the judgements represented numerically
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
          jud_res = re.findall(r"\[\[[A-E]+(?:\]\])?",jud)
          jud_set = set(jud_res)
          # Only 1 response provided
          if len(jud_res) == 1 or len(jud_set) == 1:
            res = letter_to_number[jud_res[0][2]]
          # Same as above, handling answer cutoff
          elif len(jud_set) == 2 and list(jud_set)[0][0:3]== list(jud_set)[1][0:3]:
            res = letter_to_number[jud_res[0][2]]
          # No responses provided
          elif len(jud_res) == 0:
            # Check for invalid solutions - assign these 7.0
            inv_res = re.findall(r"\[\[[A-E]+(?:\]\])?",jud)
            if len(inv_res) > 0:
              res = 7.0
            else:
              # Otherwise no response was provided
              res = 6.0
          # Duplicates listed in response format
          else:
            # Split on sentences to find response
            sent_split = re.split(r"(?<!\w\.\w\.)(?<![A-Z]\.)(?<!\w\.\w\.)\s*\.\s*", jud)
            relev_sent = [sent for sent in sent_split if 'best' in sent.lower() or 'correct' in sent.lower()]
            if len(relev_sent) == 0:
              res = 6.0
            else:
              ans = re.findall(r"\[\[[A-E]+(?:\]\])?",relev_sent[0])
              if len(set(ans)) == 1:
                res = letter_to_number[ans[0][2]]
              else:
                # Split on newlines to find response
                split_newlines = re.findall(r"(?:.*\[\[[A-E]+\]\].*)", relev_sent[0], re.MULTILINE)
                for split in split_newlines:
                  if 'best' in split.lower() or 'correct' in split.lower():
                    ans = re.findall(r"\[\[[A-E]+(?:\]\])?",split)
                    res = letter_to_number[ans[0][2]]
          curr_judgements.append(res)
      num_judge_by_mod[mod][prompt] = curr_judgements
  return num_judge_by_mod

def judge_reliab(judged_prompts_num):
  '''
  Args:
    prompts_judge_num: list of the numerical representation for the model's judgement replications for each prompt
  Returns:
    df of the reliability of the model's judgements across replications for BBH only, SQuAD only, MT-bench only, and full set of responses
  '''
  mod_print_name = mod.replace('/', '_')
  df = pd.DataFrame.from_dict(judged_prompts_num)
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
  full_reliab = reliability_analysis(raw_dataset=df, is_corr_matrix=False)
  full_reliab.fit()
  full_reliab_alph = ((full_1_reliab_count)+((df.shape[1])*full_reliab.alpha_cronbach))/(full_1_reliab_count+df.shape[1])
  full_reliab_omeg = ((full_1_reliab_count)+((df.shape[1])*full_reliab.omega_total))/(full_1_reliab_count+df.shape[1])
  bbh_df = df.loc[:, df.columns.str.startswith('bbh')]
  bbh_reliab = reliability_analysis(raw_dataset=bbh_df, is_corr_matrix=False)
  bbh_reliab.fit()
  bbh_reliab_alph = ((bbh_1_reliab_count)+((bbh_df.shape[1])*bbh_reliab.alpha_cronbach))/(bbh_1_reliab_count+bbh_df.shape[1])
  bbh_reliab_omeg = ((bbh_1_reliab_count)+((bbh_df.shape[1])*bbh_reliab.omega_total))/(bbh_1_reliab_count+bbh_df.shape[1])
  squad_df = df.loc[:, df.columns.str.startswith('squad')]
  squad_reliab = reliability_analysis(raw_dataset=squad_df, is_corr_matrix=False)
  squad_reliab.fit()
  squad_reliab_alph = ((squad_1_reliab_count)+((squad_df.shape[1])*squad_reliab.alpha_cronbach))/(squad_1_reliab_count+squad_df.shape[1])
  squad_reliab_omeg = ((squad_1_reliab_count)+((squad_df.shape[1])*squad_reliab.omega_total))/(squad_1_reliab_count+squad_df.shape[1])
  mt_df = df.loc[:, df.columns.str.startswith('mtb')]
  mt_reliab = reliability_analysis(raw_dataset=mt_df, is_corr_matrix=False)
  mt_reliab.fit()
  mt_reliab_alph = ((mt_1_reliab_count)+((mt_df.shape[1])*mt_reliab.alpha_cronbach))/(mt_1_reliab_count+mt_df.shape[1])
  mt_reliab_omeg = ((mt_1_reliab_count)+((mt_df.shape[1])*mt_reliab.omega_total))/(mt_1_reliab_count+mt_df.shape[1])
  results = {
      "alpha": [bbh_reliab_alph, squad_reliab_alph, mt_reliab_alph, full_reliab_alph],
      "omega": [bbh_reliab_omeg, squad_reliab_omeg, mt_reliab_omeg, full_reliab_omeg]
      }
  results_df = pd.DataFrame(results, index=["BBH", "SQuAD", "MTB","Full"])
  return results_df


num_judge = judge_to_numeric(judgements_by_mod)

judge_reliab_by_mod = {}
for mod, judge_dict in num_judge.items():
  judge_reliab_by_mod[mod] = judge_reliab(judge_dict)
  mod_print_name = mod.replace('/', '_')
  judge_reliab_by_mod[mod].to_csv(f'judge_reliab_{mod_print_name}.csv')
  
full_reliab_df = pd.concat([df.assign(source=key) for key, df in judge_reliab_by_mod.items()]).reset_index()
full_reliab_df = full_reliab_df.pivot_table(index='source', columns='index', values='omega').reset_index().rename_axis(None, axis=1)
full_reliab_df = full_reliab_df.rename(columns={'source': 'Model', 'Full':'All'})
new_order = ['Model','All','BBH','MTB','SQuAD']
full_reliab_df = full_reliab_df[new_order]
full_reliab_df.to_csv('judge_full_reliab.csv')
  
