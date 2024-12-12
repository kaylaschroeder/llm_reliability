import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch

from utils import generate_judge


device = "cuda:1"

# Hugging face access token
access_token = ''

# Define LLM judge types
model_types = ['Nexusflow/Starling-LM-7B-beta', 'meta-llama/Meta-Llama-3-8B-Instruct', 'google/gemma-1.1-7b-it']
               

# Load data for judgement
with open('application_responses.json', 'r') as resp:
    prompt_resp = json.load(resp)

random.seed(2)

few_shot_prefix = """You need to check whether the prediction of a question-answering systemto a question is correct. You should make the judgment based on a list of ground truth answers provided to you. Your response should be "correct" if the prediction is correct or "incorrect" if the prediction is wrong.
Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: W Shakespeare
Correctness: correct
Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: Roma Gill and W Shakespeare
Correctness: correct
Question: Who authored The Taming of the Shrew (published in 2002)?
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: Roma Shakespeare
Correctness: incorrect
Question: What country is Maharashtra Metro Rail Corporation Limited located in?
Ground truth: ["India"]
Prediction: Maharashtra
Correctness: incorrect
Question: What’s the job of Song Kang-ho in Parasite (2019)?
Ground truth: ["actor"]
Prediction: He plays the role of Kim Ki-taek, the patriarch of the Kim
family.
Correctness: correct
Question: 
"""


def model_judge(judge_model, response_model, prompts_judge, access_token):
  '''
  Args:
    judge_model: Hugging Face model name of the model used for judging
    response_model: Hugging Face model name of the model used for original Q&A responses
    prompts_judge: dictionary of prompts to be judged by the model
    access_token: Hugging Face access token
  Returns:
    list of the model's judgement replications for each prompt
  '''
  tokenizer = AutoTokenizer.from_pretrained(judge_model, token=access_token)
  model = AutoModelForCausalLM.from_pretrained(judge_model, token=access_token)

  model_judge_dict = {}
  jud_mod_print_name = judge_model.replace('/', '_')
  prompt_mod_print_name = response_model.replace('/', '_')
  for q_dict in prompts_judge['mag']:
    mod_judge = [generate_judge(model, tokenizer, few_shot_prefix+str(q_dict['question'])+'\n Ground truth: '+str(q_dict['ground_truth'])+'\n Prediction: '+ q_dict['response'][:15]+
            '\n Correctness: ',
            0.75, device) for _ in range(100)]
    model_judge_dict[q_dict['question']] = mod_judge
  for q_dict in prompts_judge['dblp']:
    mod_judge = [generate_judge(model, tokenizer, few_shot_prefix+str(q_dict['question'])+'\n Ground truth: '+str(q_dict['ground_truth'])+'\n Prediction: '+ q_dict['response'][:15]+
            '\n Correctness: ',
            0.75, device) for _ in range(100)]
    model_judge_dict[str(q_dict['question'])] = mod_judge
  with open(f'judge_full_{prompt_mod_print_name}_by_{jud_mod_print_name}.json', 'w') as f:
    json.dump(model_judge_dict, f)
  return model_judge_dict


judgements_by_mod = {}
for resp_mod, resp_mod_dict in prompt_resp.items():
    resp_mod_print_name = resp_mod.replace('/', '_')
    print(f'Judging responses from {resp_mod_print_name}')
    judgements_by_mod[resp_mod] = {}
    for jud_mod in model_types:
        jud_mod_print_name = jud_mod.replace('/', '_')
        print(f'Currently judging using {jud_mod_print_name}')
        judgements_by_mod[resp_mod][jud_mod] = model_judge(resp_mod, jud_mod, resp_mod_dict, access_token)

# Save judgements outputs to json
with open(f'application_judgements.json', 'w') as f:
    json.dump(judgements_by_mod, f)


