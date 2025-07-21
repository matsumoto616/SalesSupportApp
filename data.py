from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

class CompanyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _get_text_tensor(self, row):
        text = str(row["企業名"]) + " " + str(row["業種"]) + " " + str(row.get("備考", ""))
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids, attn_mask = self._get_text_tensor(row)
        numeric_tensor = torch.tensor([float(row['資本金(百万円)']), float(row['従業員数'])], dtype=torch.float)
        return input_ids, attn_mask, numeric_tensor
    

class TripletCompanyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _get_text_tensor(self, row):
        text = str(row["企業名"]) + " " + str(row["業種"]) + " " + str(row.get("備考", ""))
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

    def __getitem__(self, idx):
        anchor = self.df.iloc[idx]
        similar = anchor["似ていると感じる企業番号"].split(',')
        similar = [int(s) for s in similar]
        positive = self.df.iloc[similar].sample(1).iloc[0]
        different = anchor["真逆だと感じる企業番号"].split(',')
        different = [int(d) for d in different]
        negative = self.df.iloc[different].sample(1).iloc[0]

        def get_row_data(row):
            input_ids, attn_mask = self._get_text_tensor(row)
            numeric_tensor = torch.tensor([float(row['資本金(百万円)']), float(row['従業員数'])], dtype=torch.float)

            return input_ids, attn_mask, numeric_tensor

        return tuple(get_row_data(r) for r in [anchor, positive, negative])