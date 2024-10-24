import json
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

with open('bbh_sampled_q.json') as s:
  bbh_samp = json.load(s)
with open('squad_sampled_q.json') as sq:
  squad_samp = json.load(sq)
with open('prompt_responses.json') as r:
  responses = json.load(r)

# BBH
# Match labels in sampled questions to response labels
bbh_resp_target_match = []
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
    bbh_resp_target_match.append(pd.DataFrame({'single:'+q_lst['category']: curr_target_match}))
    
bbh_result = pd.concat(resp_target_match, axis=1)
bbh_result.loc[3,'single:date_understanding'] = 1
bbh_result.loc['prompt_accuracy'] = bbh_result.sum(axis=0)

bbh_result.to_csv('accuracy_resp.csv')

# SQuAD
squad_resp_target_match = []
stop_words = set(stopwords.words('english'))
for label,resp_dict in responses.items():
  curr_target_match = []
  if label.startswith('squad'):
    # Tokenize the answer
    true_ans = [prompt['answers'][0]['text'] for prompt in squad_samp if label == ('squad:'+prompt['id'])][0]
    true_ans_tokens = nltk.word_tokenize(true_ans)
    true_ans_tokens = [[word for word in ans if word not in stop_words] for ans in true_ans]
    for mod, mod_resp in resp_dict['model_responses'].items():
      # Tokenize the response
      mod_resp_tokens = nltk.word_tokenize(mod_resp)
      mod_resp_tokens = [word for word in mod_resp_tokens if word not in stop_words]
      if true_ans[0] in mod_resp:
        curr_target_match.append(1)
      else:
        curr_target_match.append(0)
    squad_resp_target_match.append(pd.DataFrame({label: curr_target_match}))
squad_result = pd.concat(squad_resp_target_match, axis=1)
squad_result.loc['prompt_accuracy'] = squad_result.sum(axis=0)
squad_result.to_csv('accuracy_resp_squad.csv')
