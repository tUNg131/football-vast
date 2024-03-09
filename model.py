import math

import torch
import torch.nn.functional as F
from torch import nn
from lightning import LightningModule

from optimizer import LAMB


class RelativePositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, dropout):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(num_embeddings, embedding_dim)

        k = torch.arange(0, num_embeddings).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[x].requires_grad_(False)


class Embedding(nn.Module):

    def __init__(self, timesteps, n_joint, d_joint, d_x, d_model, dropout):
        super().__init__()

        self.d_x = d_x

        self.linear = nn.Linear(
            in_features=d_x, out_features=d_model
        )

        self.time_embedding = nn.Embedding(
            num_embeddings=timesteps, embedding_dim=d_model
        )

        num_space_embeddings = (n_joint * d_joint) // d_x
        self.space_embedding = nn.Embedding(
            num_embeddings=num_space_embeddings, embedding_dim=d_model
        )

        self.nan_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=d_model
        )


    def forward(self, x):
        bsize, timesteps, n_joint, d_joint = x.shape

        n_token = (n_joint * d_joint) // self.d_x

        # time embedding
        time_emb = self.time_embedding(
            torch.arange(timesteps, dtype=torch.int, device=x.device)
            .view(1, -1, 1)
            .repeat(bsize, 1, n_token)
            .view(bsize, -1)
        )

        # space embedding
        space_emb = self.space_embedding(
            torch.arange(n_token, dtype=torch.int, device=x.device)
            .repeat(bsize, timesteps)
        )

        # nan embedding
        nan_emb = self.nan_embedding(
            torch.isnan(x)
            .view(bsize, -1, self.d_x)
            .any(-1)
            .int()
        )

        x = self.linear(
            torch.nan_to_num(x)
            .view(bsize, -1, self.d_x)
        )

        emb = x + time_emb + space_emb + nan_emb
        return emb


class HumanPoseModel(LightningModule):

    def __init__(self, n_timestep, n_joint, d_joint, d_x, n_heads, n_layers,
                 d_model, d_hid, dropout):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = Embedding(n_timestep, n_joint, d_joint, d_x, d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.linear = nn.Linear(d_model, d_x)

    def forward(self, x):
        emb = self.embedding(x)

        output = self.transformer_encoder(emb)

        output = self.linear(output)

        output = output.view(x.shape)

        return output

    def training_step(self, batch, batch_idx):
        data, target = batch

        data = self.add_gaussian_noise(data)

        output = self(data)
        nan_mask = torch.isnan(data)
        loss = F.mse_loss(output[nan_mask], target[nan_mask])

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        nan_mask = torch.isnan(data)
        loss = F.mse_loss(output[nan_mask], target[nan_mask])

        self.log("eval_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        nan_mask = torch.isnan(data)
        loss = F.mse_loss(output[nan_mask], target[nan_mask])

        self.log("test_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = LAMB(self.parameters(), lr=0.0015)
        return optimizer

    @staticmethod
    def add_gaussian_noise(data, std=0.02):
        noise = std * torch.randn_like(data)
        return data + noise