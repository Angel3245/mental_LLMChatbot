import torch
from pathlib import Path
from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM

#MODEL_NAME = "bigscience/bloom-petals"

path = Path.cwd()
# Cargar el modelo previamente entrenado desde el disco duro
MODEL_NAME = F"{str(path)}/output/MentalKnowledge/petals"

tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)
model = model.cuda()

inputs = tokenizer('A cat in French is '+"\n-----\n", return_tensors="pt")["input_ids"].cuda()
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))