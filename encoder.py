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
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(512, image_output_dim)

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

        # Decoder部分（AutoEncoder用）
        self.decoder = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Linear(256, image_output_dim + text_output_dim + numeric_output_dim)
        )

    def load_weights(self, path, device='cpu'):
        """
        学習済み重みをロードする（推論時用）
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()

    def forward(self, image, input_ids, attention_mask, numeric):
        # Encoder
        img_feat = self.image_encoder(image).view(image.size(0), -1)
        img_vec = self.image_fc(img_feat)

        txt_feat = self.text_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        txt_vec = self.text_fc(txt_feat)

        num_vec = self.numeric_encoder(numeric)

        fused = torch.cat([img_vec, txt_vec, num_vec], dim=1)
        z = self.fusion(fused)

        # Decoder（再構成）
        recon = self.decoder(z)

        # それぞれのベクトルも返す
        return z, recon, fused

def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()


def train_encoder(model, dataset, epochs=5, batch_size=8, lr=1e-4, device='cpu'):
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            # batch: ((img_a, id_a, mask_a, num_a), (img_p, id_p, mask_p, num_p), (img_n, id_n, mask_n, num_n))
            (img_a, id_a, mask_a, num_a), (img_p, id_p, mask_p, num_p), (img_n, id_n, mask_n, num_n) = batch
            img_a, id_a, mask_a, num_a = img_a.to(device), id_a.to(device), mask_a.to(device), num_a.to(device)
            img_p, id_p, mask_p, num_p = img_p.to(device), id_p.to(device), mask_p.to(device), num_p.to(device)
            img_n, id_n, mask_n, num_n = img_n.to(device), id_n.to(device), mask_n.to(device), num_n.to(device)

            # AutoEncoder loss for anchor
            z_a, recon_a, orig_a = model(img_a, id_a, mask_a, num_a)
            ae_loss = F.mse_loss(recon_a, orig_a)

            # Triplet loss (unsupervised distance learning)
            z_p, _, _, = model(img_p, id_p, mask_p, num_p)
            z_n, _, _, = model(img_n, id_n, mask_n, num_n)
            trip_loss = triplet_loss(z_a, z_p, z_n)

            loss = ae_loss + trip_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")

    return model


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("./db/companies_archive.csv")  # Load your dataset here
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = TripletCompanyDataset(df, tokenizer)

    model = Encoder()
    trained_model = train_encoder(model, dataset, epochs=10, batch_size=8, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "./weights/encoder.pth")