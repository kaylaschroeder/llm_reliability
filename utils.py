from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
import torch

def generate_txt(model, tokenizer, question, seed_val):

  set_seed(seed_val)
  generated_tensor = model.generate(**tokenizer(question, return_tensors="pt").to(device),
                                      pad_token_id=tokenizer.eos_token_id)
  generated_text = tokenizer.decode(generated_tensor[0])
  answer = generated_text.replace(question, "").strip()
  return answer
