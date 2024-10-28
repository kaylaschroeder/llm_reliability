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
    df of the reliability of the model's judgements across replications for BBH only, MT-bench only, and full set of responses
  '''
  mod_print_name = mod.replace('/', '_')
  df = pd.DataFrame.from_dict(judged_prompts_num)
  # Drop columns where LLM makes no judgements
  for col in df.columns:
    if len(df[col].unique()) == 1 and df[col].unique() == [6.0]:
      df.drop(col, axis=1, inplace=True)
  full_reliab = reliability_analysis(raw_dataset=df, is_corr_matrix=False)
  full_reliab.fit()
  bbh_df = df.loc[:, df.columns.str.startswith('single')]
  bbh_reliab = reliability_analysis(raw_dataset=bbh_df, is_corr_matrix=False)
  bbh_reliab.fit()
  mt_df = df.loc[:, df.columns.str.startswith('multi')]
  mt_reliab = reliability_analysis(raw_dataset=mt_df, is_corr_matrix=False)
  mt_reliab.fit()
  results = {
      "alpha": [bbh_reliab.alpha_cronbach, mt_reliab.alpha_cronbach, full_reliab.alpha_cronbach],
      "omega": [bbh_reliab.omega_total, mt_reliab.omega_total, full_reliab.omega_total]
      }
  results_df = pd.DataFrame(results, index=["BBH", "MTB","Full"])
  return results_df


num_judge = judge_to_numeric(judgements_by_mod)

judge_reliab_by_mod = {}
for mod, judge_dict in num_judge.items():
  judge_reliab_by_mod[mod] = judge_reliab(judge_dict)
  mod_print_name = mod.replace('/', '_')
  judge_reliab_by_mod[mod].to_csv(f'judge_reliab_{mod_print_name}.csv')
  
