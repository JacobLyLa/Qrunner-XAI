import os
import re

import numpy as np
import torch
import torch.nn as nn


# Architecture inspired by:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L30
# And original DQN paper
class QNetwork(nn.Module):
    def __init__(self, colors=3, actions=5, model_path=None):
        super().__init__()
        self.colors = colors

        self.network = nn.Sequential(
            nn.Conv2d(colors, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, actions)
        )

        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
            parts = model_path.split("/")
            run_id = parts[1]
            model_part = parts[2].split("_")[1].split(".")[0]
            self.model_name = f"{run_id}-{model_part}"
            print(f"Loaded model: {self.model_name}")

    def forward(self, x, return_acts=False):
        # Note return_acts detachs the activations from the graph, can't directly backprop through them
        if x.shape[1] == x.shape[2]:
            # Change to (batch, channel, width, height)
            x = x.permute(0, 3, 1, 2)  

        x = x / 255.0
        activations = {}
        for idx, layer in enumerate(self.network.children()):
            x = layer(x)
            if return_acts and idx < len(self.network) - 1:
                activations[idx] = x.clone().detach()
        if return_acts:
            return x, activations
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def load_qnetwork_device(model_path="runs/20240118-143135/model_1000000.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qnetwork = QNetwork(model_path=model_path).to(device).eval()
        return qnetwork, device
    
    @staticmethod
    def find_newest_model():
        base_path = 'runs'
        newest_dir = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d != "sb3"], 
                            key=lambda x: x.split('/')[-1], reverse=True)[0]


        # Find model with highest steps
        model_files = [f for f in os.listdir(os.path.join(base_path, newest_dir)) if f.endswith('.pt')]
        newest_model = sorted(model_files, key=lambda x: int(re.search('model_(\d+).pt', x).group(1)), reverse=True)[0]

        return os.path.join(base_path, newest_dir, newest_model)

if __name__ == '__main__':
    batch_size = 32
    batch = torch.rand((batch_size, 84, 84, 3))
    model = QNetwork()
    print(model)
    print(f"Number of parameters in the model: {model.count_parameters()}")
    output = model(batch)
    print(output.shape)