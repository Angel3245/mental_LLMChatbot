import torch
import csv, sys
from pathlib import Path
from transformers import BloomTokenizerFast 
from petals import DistributedBloomForCausalLM

MODEL_NAME = "bigscience/bloom-petals"

path = Path.cwd()
# Cargar el modelo previamente entrenado desde el disco duro
#MODEL_NAME = F"{str(path)}/output/MentalKnowledge/petals"

tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
model = DistributedBloomForCausalLM.from_pretrained(MODEL_NAME)

prompt = "What is depression?"

route_response = F"{str(path)}/file/prompt_test/prompt.csv"

response = ""

try:
    with model.inference_session(max_length=512) as sess:
        print('Prompt', file=sys.stderr)
        print(prompt, file=sys.stderr)
        prefix = tokenizer(prompt, return_tensors="pt")["input_ids"]
        
        while True:
            outputs = model.generate(
                prefix, max_new_tokens=1, do_sample=True, top_p=0.9, temperature=0.75, session=sess
            )
            outputs = tokenizer.decode(outputs[0, -1:])
            response += outputs
            print(response, file=sys.stderr)

            prefix = None

except Exception as e:
    print(e, file=sys.stderr)

with open(route_response, 'w', newline="", encoding="UTF8") as csv_file:
    writer = csv.writer(csv_file, delimiter=",")
    writer.writerow([response])