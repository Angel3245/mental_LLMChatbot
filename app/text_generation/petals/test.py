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

inputs = tokenizer('The following is a conversation with a mental health expert. Expert helps the User by providing emotional support, it also helps solving doubts related to mental health by providing the best option. If the expert does not know the answer to a question, it truthfully says it does not know. The expert is conversational, optimistic, flexible, empathetic, creative and humanly in generating responses.\nUser: What is mental health?\nBot: '+"\n-----\n", return_tensors="pt")["input_ids"].cuda()
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))