import torch
import torch.nn as nn

# Architecture from:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L30

class QNetwork(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))

    def forward(self, x, return_acts=False):
        x = x / 255.0
        activations = {}
        for idx, (name, layer) in enumerate(self.network.named_children()):
            x = layer(x)
            if return_acts and not isinstance(layer, nn.Flatten) and idx < len(self.network) - 1:
                activations[name] = x.clone()
        if return_acts:
            return x, activations
        return x