import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_channels=3, actions=5, use_dueling=False):
        super().__init__()
        self.input_channels = input_channels
        self.use_dueling = use_dueling

        """
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, 8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ReLU(),
            nn.Flatten()
        )
        """

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, bias=False), # 84x84 -> 82x82
            nn.ReLU(),
            nn.MaxPool2d(2), # 82x82 -> 41x41
            nn.Conv2d(64, 64, 3, bias=False), # 41x41 -> 39x39
            nn.ReLU(),
            nn.MaxPool2d(2), # 39x39 -> 19x19
            nn.Conv2d(64, 64, 3, bias=False), # 19x19 -> 17x17
            nn.ReLU(),
            nn.MaxPool2d(2), # 17x17 -> 8x8
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )

        if self.use_dueling:
            # Dueling DQN: Separate streams for Value and Advantage
            self.value_stream = nn.Sequential(
                nn.Linear(128, 1),
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(128, actions),
            )
        else:
            # Standard DQN: Single output stream
            self.output = nn.Linear(128, actions)

    def forward(self, x):
        x = self.feature_extractor(x / 255.0)
        x = self.fc(x)
        
        if self.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Combine value and advantage streams to get final Q-values
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            return self.output(x)

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