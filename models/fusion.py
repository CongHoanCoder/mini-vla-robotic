"""Fusion module to combine image, text, and state tokens."""

import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, img_token, txt_token, state_token):
        x = torch.cat([img_token, txt_token, state_token], dim=-1)
        x = self.net(x)
        x = self.ln(x)
        return x  # fused context


class FusionTransformer(nn.Module):
    """Cross-modal fusion via a transformer encoder.

    Keeps image/text/state as separate tokens so they can attend to each
    other, then aggregates into a single fused context via a learned [CLS]
    token. Same I/O contract as FusionMLP.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, img_token, txt_token, state_token):
        # each token: (B, d_model)
        cls_tokens = self.cls_token.expand(img_token.size(0), -1, -1)     # (B, 1, d_model)
        tokens = torch.stack([img_token, txt_token, state_token], dim=1)  # (B, 3, d_model)
        tokens = torch.cat([cls_tokens, tokens], dim=1)                   # (B, 4, d_model)
        tokens = self.transformer(tokens)                                 # (B, 4, d_model)
        return self.ln(tokens[:, 0])  # (B, d_model) fused context
