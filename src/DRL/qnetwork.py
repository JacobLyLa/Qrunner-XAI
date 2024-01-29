import torch
import torch.nn as nn

'''
# Architecture inspired from:
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py#L30

# Expects input of shape (batch_size, frame_stacks*colors, 84, 84)
# Outputs of shape (batch_size, actions)
# TODO: Try 3d convolutions (batch_size, frame_stacks, colors, 84, 84)
class QNetwork(nn.Module):
    def __init__(self, frame_stacks, colors=3, actions=5, model_path=None):
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
        # Check if channels are last (batch, width, height, channel)
        if x.shape[1] == x.shape[2]:
            x = x.permute(0, 3, 1, 2)  # Change to (batch, channel, width, height)

        x = x / 255.0
        activations = {}
        for idx, (name, layer) in enumerate(self.network.named_children()):
            x = layer(x)
            if return_acts and idx < len(self.network) - 1:
                activations[name] = x.clone().detach()
        if return_acts:
            return x, activations
        return x
    
    def load_qnetwork_device(model_path="runs/20240118-143135/model_1000000.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qnetwork = QNetwork(model_path).to(device).eval()
        return q_network, device
'''

class QNetwork(nn.Module):
    def __init__(self, frame_stacks, use_3d=False, colors=3, actions=5, model_path=None):
        super().__init__()
        self.frame_stacks = frame_stacks
        self.colors = colors
        self.use_3d = use_3d

        if use_3d:
            self.network = nn.Sequential(
                nn.Conv3d(colors, 32, (2, 8, 8), stride=(1, 4, 4)),
                nn.ReLU(),
                nn.Conv3d(32, 64, (1, 4, 4), stride=(1, 2, 2)),
                nn.ReLU(),
                nn.Conv3d(64, 64, (1, 3, 3), stride=(1, 1, 1)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 2 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, actions)
            )
        else:
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
        if x.shape[1] == x.shape[2]:
            # Change to (batch, channel, width, height)
            x = x.permute(0, 3, 1, 2)  

        if self.use_3d:
            x = x.view(-1, self.colors, self.frame_stacks, x.shape[2], x.shape[3])

        x = x / 255.0
        activations = {}
        #print("Shapes at each layer:")
        for idx, (name, layer) in enumerate(self.network.named_children()):
            #print(name, x.shape)
            x = layer(x)
            if return_acts and idx < len(self.network) - 1:
                activations[name] = x.clone().detach()
        if return_acts:
            return x, activations
        return x
    
    @staticmethod
    def load_qnetwork_device(model_path="runs/20240118-143135/model_1000000.pt", use_3d=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qnetwork = QNetwork(model_path=model_path, use_3d=use_3d).to(device).eval()
        return qnetwork, device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    batch_size = 32
    batch = torch.rand((batch_size, 84, 84, 12))
    use_3d = True
    
    print(f"Using 3D convolutions: {use_3d}")
    model = QNetwork(frame_stacks=4, use_3d=use_3d)
    print(model)
    print(f"Number of parameters in the model: {count_parameters(model)}")
    output = model(batch)
    print(output.shape)