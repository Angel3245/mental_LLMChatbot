import torch
from torch.utils.data import Dataset

class LlamaDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data[idx]['prompt'])
        label_text = str(self.data[idx]['completion'])

        # Add special tokens to input and label text
        text = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + input_text + "\n\n### Response:\n" + label_text

        # Tokenize input and label text
        input_encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)

        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(input_encodings['input_ids']),
        }

