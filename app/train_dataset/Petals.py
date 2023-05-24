import torch
from torch.utils.data import Dataset

class PetalsDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = str(self.data[idx]['prompt'])
        label_text = str(self.data[idx]['completion'])

        # Add special tokens to input and label text
        text = "User: " + input_text + "\nBot: " + label_text + "\n"

        # Using with open()
        #with open('log.txt', 'a') as f:
        #    print(text, file=f)

        # Tokenize input and label text
        input_encodings = self.tokenizer(text, truncation=True, padding='max_length')

        return {
            'input_ids': torch.tensor(input_encodings['input_ids']),
            'labels': torch.tensor(input_encodings['input_ids']),
        }

