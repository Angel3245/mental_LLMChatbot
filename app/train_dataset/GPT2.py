import torch
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data[idx]['input_text'])
        label_text = str(self.data[idx]['label_text'])

        # Add special tokens to input and label text
        text = self.tokenizer.bos_token + input_text + self.tokenizer.sep_token + label_text + self.tokenizer.eos_token

        # Tokenize input and label text
        input_encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=512)

        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'attention_mask': torch.tensor(input_encodings['attention_mask']),
            'labels': torch.tensor(input_encodings['input_ids']),
        }

