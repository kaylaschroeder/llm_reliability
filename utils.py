from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch

def generate_txt(model, tokenizer, question, seed_val, device):
  """
  Generates text using the given model, tokenizer, question, and seed value.
  
  Args:
    model (transformers.PreTrainedModel): The pre-trained language model.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    question (str): The question to generate a response for.
    seed_val (int): The random seed value.

  Returns:
    str: The generated text response.
  """
  set_seed(seed_val)
  model.to(device)
  inputs = tokenizer(question, return_tensors="pt").to(device)
  generated_tensor = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature = 0.75, top_k=50)
  generated_tensor = generated_tensor.to(device)
  generated_text = tokenizer.decode(generated_tensor[0], skip_special_tokens=True)
  answer = generated_text.replace(question, "").strip()
  return answer

def generate_judge(model, tokenizer, question, temp, device):
  """
  Generates judgement using the given model, tokenizer, question, and seed value.
  
  Args:
    model (transformers.PreTrainedModel): The pre-trained language model.
    tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
    question (str): The question to generate a response for.
    temp (float): The value specified for the temperature parameter.

  Returns:
    str: The generated text response.
  """
  model.to(device)
  inputs = tokenizer(question, return_tensors="pt").to(device)
  generated_tensor = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature = temp, top_k=50)
  generated_tensor = generated_tensor.to(device)
  generated_text = tokenizer.decode(generated_tensor[0], skip_special_tokens=True)
  answer = generated_text.replace(question, "").strip()
  return answer


