from reliabilipy import reliability_analysis
import pandas as pd

model_types = ['meta-llama/Meta-Llama-3-8B',
#               'mistralai/Mistral-7B-v0.1',
               'lmsys/vicuna-7b-v1.5',
               'google/gemma-7b',
               'bigscience/bloom-7b1',
               'microsoft/phi-2',
               'tiiuae/falcon-7b']

# Load data judgement data
with open('full_judgements.json', 'r') as resp:
        judgements_by_mod = json.load(resp)

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
  results_df.to_csv(f'judge_reliab_{mod_name}.csv')
  return results_df
  
  
reliability_by_mod = {}
for mod in model_types:
    reliability_by_mod[mod] = judge_reliab(mod, judgements_by_mod[mod])
    

# Combine reliability into full df for all models
combined_df = pd.concat([df.assign(key=key) for key, df in reliability_by_mod.items()], ignore_index=True)
combined_df.to_csv('full_judge_reliab.csv')
  
