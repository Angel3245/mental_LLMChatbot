import torch
from torch.utils.data import Dataset
from shared.prompter import Prompter

class LlamaDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data[idx]['prompt'])
        label_text = str(self.data[idx]['completion'])

        prompter = Prompter("alpaca")

        # Generate prompt from template
        text = prompter.generate_prompt("Answer as a mental health expert.",input_text,label_text)

        # Add stop token
        text += self.tokenizer.eos_token

        # Tokenize input and label text
        input_encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)

        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(input_encodings['input_ids']),
        }

