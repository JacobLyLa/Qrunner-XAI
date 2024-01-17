import torch
import torch.nn as nn

# Architecture inspired from:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L30

# Expects input of shape (batch_size, frame_stacks*frame_stacks, 84, 84)
# Outputs of shape (batch_size, actions)
class QNetwork(nn.Module):
    def __init__(self, frame_stacks=4, colors=3, actions=5, model_path=None):
        super().__init__()
        # (4 frames * 3 color channels each)
        self.network = nn.Sequential(
            nn.Conv2d(frame_stacks * colors, 32, 8, stride=4),
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

    def forward(self, x, return_acts=False):
        x = x / 255.0
        activations = {}
        for idx, (name, layer) in enumerate(self.network.named_children()):
            print(x.shape)
            x = layer(x)
            if return_acts and idx < len(self.network) - 1:
                activations[name] = x.clone().detach()
        if return_acts:
            return x, activations
        return x

if __name__ == '__main__':
    model = QNetwork(frame_stacks=4, colors=3, actions=5)
    print(model)
    batch_size = 32
    batch = torch.rand((batch_size, 12, 84, 84))
    output = model(batch)
    print(output.shape)
