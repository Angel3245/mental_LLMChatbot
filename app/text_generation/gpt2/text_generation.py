import torch
from transformers import GenerationConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from shared.prompter import Prompter

class GPT2Chatbot:
    def __init__(self, model_path, template="mentalbot"):
        self.model_path = model_path

        self.cutoff_len = 512

        self.prompter = Prompter(template)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Allow batched inference

        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, input_text, max_length=1000, top_p=0.9):
    
        self.model = self.model.eval()

        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)
        attention_mask = input_encodings['attention_mask'].to(self.model.device)

        # Use model.generate() to generate the response
        response = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            top_p=top_p,
            do_sample=True,
        )

        # Decode the response from the model back into text
        decoded_output = self.tokenizer.decode(response[0][ : -1])
        response = self.prompter.get_response(decoded_output)

        return response
