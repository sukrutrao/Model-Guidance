import torch.nn as nn


class LogitBias(nn.Module):
    def __init__(self, logit_bias):
        super().__init__()
        self.logit_bias = logit_bias

    def forward(self, in_tensor):
        return in_tensor + self.logit_bias

    def extra_repr(self) -> str:
        return f"logit_bias={self.logit_bias}"
