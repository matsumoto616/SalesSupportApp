import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from transformers import AutoModel, AutoTokenizer
import pandas as pd

from data import TripletCompanyDataset


class Encoder(nn.Module):
    def __init__(self, text_model="distilbert-base-uncased", 
                 image_output_dim=128, text_output_dim=128, numeric_output_dim=64, final_dim=16):
        super().__init__()

        # Encoder部分
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # self.image_fc = nn.Linear(512, image_output_dim)
        image_output_dim = 0

        self.text_bert = AutoModel.from_pretrained(text_model)
        self.text_fc = nn.Linear(self.text_bert.config.hidden_size, text_output_dim)

        self.numeric_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, numeric_output_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(image_output_dim + text_output_dim + numeric_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, final_dim)
        )

    def load_weights(self, path, device='cpu'):
        """
        学習済み重みをロードする（推論時用）
        """
        self.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        self.eval()

    def forward(self, input_ids, attention_mask, numeric):
        # Encoder
        # img_feat = self.image_encoder(image).view(image.size(0), -1)
        # img_vec = self.image_fc(img_feat)

        txt_feat = self.text_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        txt_vec = self.text_fc(txt_feat)

        num_vec = self.numeric_encoder(numeric)

        # fused = torch.cat([img_vec, txt_vec, num_vec], dim=1)
        fused = torch.cat([txt_vec, num_vec], dim=1)
        z = self.fusion(fused)

        # それぞれのベクトルも返す
        return z, fused

def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()


def train_encoder(model, dataset, epochs=5, batch_size=8, lr=1e-4, device='cpu', test_ratio=0.2):
    from torch.utils.data import random_split, DataLoader
    model = model.to(device)
    # データ分割
    n_total = len(dataset)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float('inf')
    best_weights = None
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            (id_a, mask_a, num_a), (id_p, mask_p, num_p), (id_n, mask_n, num_n) = batch
            id_a, mask_a, num_a = id_a.to(device), mask_a.to(device), num_a.to(device)
            id_p, mask_p, num_p = id_p.to(device), mask_p.to(device), num_p.to(device)
            id_n, mask_n, num_n = id_n.to(device), mask_n.to(device), num_n.to(device)

            z_a, _, = model(id_a, mask_a, num_a)
            z_p, _, = model(id_p, mask_p, num_p)
            z_n, _, = model(id_n, mask_n, num_n)
            trip_loss = triplet_loss(z_a, z_p, z_n)
            loss = trip_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Evaluation ---
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                (id_a, mask_a, num_a), (id_p, mask_p, num_p), (id_n, mask_n, num_n) = batch
                id_a, mask_a, num_a = id_a.to(device), mask_a.to(device), num_a.to(device)
                id_p, mask_p, num_p = id_p.to(device), mask_p.to(device), num_p.to(device)
                id_n, mask_n, num_n = id_n.to(device), mask_n.to(device), num_n.to(device)

                z_a, _, = model(id_a, mask_a, num_a)
                z_p, _, = model(id_p, mask_p, num_p)
                z_n, _, = model(id_n, mask_n, num_n)
                trip_loss = triplet_loss(z_a, z_p, z_n)
                loss = trip_loss
                test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        # ベストな重みを保存
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_weights = model.state_dict()
            torch.save(best_weights, "./weights/triplet_encoder_best.pth")

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("./db/companies_archive_rev.csv")  # Load your dataset here
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = TripletCompanyDataset(df, tokenizer)

    model = Encoder()
    train_encoder(model, dataset, epochs=50, batch_size=8, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu')