import json
import pandas as pd
import os
import random

# MT-bench
mt_bench_questions = []
with open('benchmarks/mt_bench.json', 'r') as mt_bench_file:
    for line in mt_bench_file:
        try:
            mt_bench_json = json.loads(line)
            mt_bench_questions.append(mt_bench_json)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
df_mt_bench = pd.DataFrame(mt_bench_questions)
mt_bench_sampled_questions = df_mt_bench.groupby('category').sample(n=1, random_state=1)
mt_bench_file = 'mt_bench_sampled_q.json'
with open(mt_bench_file, 'w') as f:
  json.dump(mt_bench_sampled_questions[['category','turns']].to_dict(orient='records'), f, indent=4)
  
# BBH
def extract_rand_element_bbh(folder_path):
    extracted_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), 'r') as f:
                try:
                    data = json.load(f)
                    if 'examples' in data:
                        random_element = random.choice(data['examples'])
                        extracted_data.append((filename[:-5], random_element['input'],random_element['target']))
                except json.JSONDecodeError:
                    print(f"Skipping {filename} due to invalid JSON format.")
    return extracted_data

random.seed(1)
bbh_data_sampled = extract_rand_element_bbh('benchmarks/big_bench_hard')
df_bbh_sampled = pd.DataFrame(bbh_data_sampled,columns=['category','input','target'])
bbh_file = 'bbh_sampled_q.json'
with open(bbh_file, 'w') as f:
  json.dump(df_bbh_sampled.to_dict(orient='records'), f, indent=4)
  
# SQuAD
with open('squad_qs.json','r') as f:
    squad_qs = json.load(f)
random.seed(1)
sampled_dat = random.sample(squad_qs['data'],20)
sampled_paragraphs = [random.sample(dat['paragraphs'],1)[0] for dat in sampled_dat]
squad_sampled_qs = []
for par in sampled_paragraphs:
    curr_dict = {}
    curr_dict['context'] = par['context']
    q_dict = par['qas'][0]
    curr_dict['question'] = q_dict['question']
    curr_dict['answers'] = q_dict['answers']
    curr_dict['is_impossible'] = q_dict['is_impossible']
    curr_dict['id'] = q_dict['id']
    squad_sampled_qs.append(curr_dict)

with open('squad_sampled_q.json','w') as f:
    json.dump(squad_sampled_qs,f)
