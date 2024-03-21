import torch
from torch import nn, Tensor


class Embedding(nn.Module):

    def __init__(self,
                 n_timesteps: int,
                 n_joints: int,
                 d_joint: int,
                 d_model: int) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features=d_joint,
                                out_features=d_model)

        self.time_embedding = nn.Embedding(num_embeddings=n_timesteps,
                                           embedding_dim=d_model)

        self.joint_embedding = nn.Embedding(num_embeddings=n_joints,
                                            embedding_dim=d_model)

        self.nan_embedding = nn.Embedding(num_embeddings=2,
                                          embedding_dim=d_model)

        self.register_buffer(
            "timestep_labels",
            torch.arange(n_timesteps, dtype=torch.int).repeat_interleave(n_joints),
            persistent=False
        )

        self.register_buffer(
            "joint_labels",
            torch.arange(n_joints, dtype=torch.int).repeat(n_timesteps),
            persistent=False
        )

    def forward(self, x: Tensor):
        batch_size, *_ = x.shape

        time_emb = self.time_embedding(self.timestep_labels.repeat(batch_size, 1))

        joint_emb = self.joint_embedding(self.joint_labels.repeat(batch_size, 1))

        nan_emb = self.nan_embedding(x.isnan().any(dim=-1, keepdim=False).int())

        val_emb = self.linear(x.nan_to_num(nan=0.0))

        return val_emb + time_emb + joint_emb + nan_emb


class FootballTransformer(nn.Module):

    def __init__(self,
                 n_timesteps: int,
                 n_joints: int,
                 d_joint: int,
                 n_heads: int,
                 n_layers: int,
                 d_model: int,
                 d_feedforward: int,
                 dropout: int,
                 activation: int) -> None:
        super().__init__()

        self.embedding = Embedding(n_timesteps=n_timesteps,
                                   n_joints=n_joints,
                                   d_joint=d_joint,
                                   d_model=d_model)
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True
            ),
            num_layers=n_layers,
        )

        self.linear = nn.Linear(in_features=d_model,
                                out_features=d_joint)
        
        self.n_timesteps = n_timesteps
        self.n_joints = n_joints
        self.d_joint = d_joint

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.n_timesteps * self.n_joints, self.d_joint)

        embed = self.embedding(x)
        feature = self.encoder(embed)
        output = self.linear(feature)

        output = output.view(-1, self.n_timesteps, self.n_joints, self.d_joint)
        return output