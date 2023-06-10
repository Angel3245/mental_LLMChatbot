import openai
import csv, evaluate
from shared import make_dirs
from shared.prompter import Prompter

class GPT3:

    def __init__(self, model, template="gpt3"):
        self.model = model

        self.prompter = Prompter(template)

    def evaluation(self, test_dataset, output_path):
        
        test_inputs = test_dataset["train"]

        # Load metrics
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")

        # Create CSV with evaluation results
        make_dirs(output_path)
        with open(output_path+"/evaluation.csv", 'w', encoding="UTF8") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["Input","Response","Bleu-1","Rouge-1"])

        for input_text in test_inputs:
            response = self.generate_response(input_text["input"])

            # Create CSV with evaluation results
            with open(output_path+"/evaluation.csv", 'a', encoding="UTF8") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow([input_text["input"],response, round(bleu_metric.compute(predictions=[response],references=[input_text["output_expected"]])['precisions'][0] ,2), round(rouge_metric.compute(predictions=[response],references=[input_text["output_expected"]])['rouge1'] ,2)])

    def generate_response(self, input_text):
        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        return openai.Completion.create(
            model=self.model,
            prompt=prompt)