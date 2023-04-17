import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Chatbot:
    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, input_text, max_length=30, top_p=0.9):
        input_encodings = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.device)
        attention_mask = input_encodings['attention_mask'].to(self.device)

        # Use model.generate() to generate the response
        response = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            top_p=top_p,
            do_sample=True,
        )

        # Decode the response from the model back into text
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)

        return response_text
