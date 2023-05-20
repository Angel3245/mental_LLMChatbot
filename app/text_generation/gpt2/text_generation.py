import torch
from shared.model_classes import MODEL_CLASSES
from transformers import GenerationConfig


class GPT2Chatbot:
    def __init__(self, model_name, model_path):
        self.model_path = model_path
        model_class, tokenizer_class, model_name_or_path = MODEL_CLASSES[model_name]

        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Allow batched inference
        self.tokenizer.sep_token = "<sep>"

        self.model = model_class.from_pretrained(model_path)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, input_text, max_length=1000, top_p=0.9):
        input_text = self.tokenizer.bos_token + input_text + self.tokenizer.sep_token

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

        # Remove eos_token and decode the response from the model back into text
        response_text = self.tokenizer.decode(response[0][ : -1], skip_special_tokens=False).split(self.tokenizer.sep_token)[1]

        return response_text
