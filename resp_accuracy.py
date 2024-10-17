import json
import pandas as pd
import re

with open('bbh_sampled_q.json') as s:
  bbh_samp = json.load(s)
with open('prompt_responses.json') as r:
  responses = json.load(r)

# Match labels in sampled questions to response labels
resp_target_match = []
for q_lst in bbh_samp:
  curr_target_match = []
  target = q_lst['target']
  # Only multiple choice questions
  pattern = r'^\(\w\)$'
  if re.match(pattern, target) is not None:
    curr_resps = responses['single:'+q_lst['category']]['model_responses']
    targ_re = re.escape(target)
    for resp in curr_resps.values():
      if target in resp:
        all_matches = re.findall(r"([^.]*?"+targ_re+r"[^.]*\.)",resp)
        if all_matches is not None:
          incorrect_bool = False
          q_repeated_count = 0
          for mat in all_matches:
            if 'incorrect' in mat:
              incorrect_bool = True
            if q_lst['input'][-75:] in mat:
              q_repeated_count += 1
          if incorrect_bool == False and q_repeated_count < len(all_matches):
            curr_target_match.append(1)
          else:
            curr_target_match.append(0)
      else:
        curr_target_match.append(0)
    resp_target_match.append(pd.DataFrame({'single:'+q_lst['category']: curr_target_match}))
    
result = pd.concat(resp_target_match, axis=1)
result.loc[3,'single:date_understanding'] = 1
result.loc['prompt_accuracy'] = result.sum(axis=0)

result.to_csv('accuracy_resp.csv')
    
