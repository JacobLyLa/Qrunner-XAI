import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels=3, actions=5, use_dueling=False):
        super().__init__()
        self.input_channels = input_channels
        self.use_dueling = use_dueling
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU()
        )
        
        if self.use_dueling:
            # Dueling DQN: Separate streams for Value and Advantage
            self.value_stream = nn.Sequential(
                nn.Linear(64, 1),
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(64, actions),
            )
        else:
            # Standard DQN: Single output stream
            self.output = nn.Linear(64, actions)

    def forward(self, x, return_acts=False):
        # Note: return_acts detaches the activations from the graph; can't directly backprop through them
        activations = {}
        layer_idx = 0  # To keep track of layer indices for activation storage

        # Normalize input
        x = x / 255.0

        # Iterate through feature extractor layers
        for layer in self.feature_extractor:
            x = layer(x)
            if return_acts:
                activations[layer_idx] = x.clone().detach()
            layer_idx += 1

        # Pass through fully connected layers
        for layer in self.fc:
            x = layer(x)
            if return_acts:
                activations[layer_idx] = x.clone().detach()
            layer_idx += 1

        if self.use_dueling:
            # Pass through value stream
            value = self.value_stream(x)
            if return_acts:
                activations[layer_idx] = value.clone().detach()
            layer_idx += 1

            # Pass through advantage stream
            advantage = self.advantage_stream(x)
            if return_acts:
                activations[layer_idx] = advantage.clone().detach()
            layer_idx += 1

            # Combine value and advantage streams to get final Q-values
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard DQN: Single output stream
            q_values = self.output(x)
            if return_acts:
                activations[layer_idx] = q_values.clone().detach()
            layer_idx += 1

        if return_acts:
            return q_values, activations
        return q_values

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == '__main__':
    import torch

    batch_size = 32
    batch = torch.rand((batch_size, 3, 84, 84))  # Use 1 for grayscale
    model = DQN(input_channels=3, use_dueling=False)  # Initialize with grayscale channels and dueling architecture
    print(model)
    print(f"Number of parameters in the model: {model.count_parameters()}")
    output = model(batch)
    print(output.shape)
    
    load_path = "models/qrunner_dqn_20241031_183101.pth"
    model = torch.load(load_path)
    print(model.keys())
    first_weights = model['feature_extractor.0.weight'].cpu().numpy()
    print(first_weights.shape)