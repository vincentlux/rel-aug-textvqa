# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from src.common.registry import registry
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class LanguageDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.language_lstm = nn.LSTMCell(
            in_dim + kwargs["hidden_dim"], kwargs["hidden_dim"], bias=True
        )
        self.fc = weight_norm(nn.Linear(kwargs["hidden_dim"], out_dim))
        self.dropout = nn.Dropout(p=kwargs["dropout"])
        self.init_weights(kwargs["fc_bias_init"])

    def init_weights(self, fc_bias_init):
        self.fc.bias.data.fill_(fc_bias_init)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, weighted_attn):
        # Get LSTM state
        state = registry.get(f"{weighted_attn.device}_lstm_state")
        h1, c1 = state["td_hidden"]
        h2, c2 = state["lm_hidden"]

        # Language LSTM
        h2, c2 = self.language_lstm(torch.cat([weighted_attn, h1], dim=1), (h2, c2))
        predictions = self.fc(self.dropout(h2))

        # Update hidden state for t+1
        state["lm_hidden"] = (h2, c2)

        return predictions
