from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

class CompanyDataset(Dataset):
    def __init__(self, df, tokenizer, image_size=224, max_len=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _get_text_tensor(self, row):
        text = str(row["企業名"]) + " " + str(row["業種"]) + " " + str(row.get("備考", ""))
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

    def _get_image_tensor(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids, attn_mask = self._get_text_tensor(row)
        image_tensor = self._get_image_tensor(row['画像パス']) if '画像パス' in row else torch.zeros(3, 224, 224)
        numeric_tensor = torch.tensor([float(row['資本金(百万円)']), float(row['従業員数'])], dtype=torch.float)
        return image_tensor, input_ids, attn_mask, numeric_tensor
    

class TripletCompanyDataset(Dataset):
    def __init__(self, df, tokenizer, image_size=224, max_len=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _get_text_tensor(self, row):
        text = str(row["企業名"]) + " " + str(row["業種"]) + " " + str(row.get("備考", ""))
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

    def _get_image_tensor(self, path):
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        anchor = self.df.iloc[idx]
        same_industry = self.df[self.df['業種'] == anchor['業種']]
        positive = same_industry.sample(1).iloc[0] if len(same_industry) > 1 else anchor
        different = self.df[self.df['業種'] != anchor['業種']]
        negative = different.sample(1).iloc[0] if len(different) > 0 else anchor

        def get_row_data(row):
            input_ids, attn_mask = self._get_text_tensor(row)
            image_tensor = self._get_image_tensor(row['画像パス']) if '画像パス' in row else torch.zeros(3, 224, 224)
            numeric_tensor = torch.tensor([float(row['資本金(百万円)']), float(row['従業員数'])], dtype=torch.float)
            return image_tensor, input_ids, attn_mask, numeric_tensor

        return tuple(get_row_data(r) for r in [anchor, positive, negative])