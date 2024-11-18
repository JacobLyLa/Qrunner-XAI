import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, colors=3, actions=5, model_path=None):
        super(ActorCriticNetwork, self).__init__()
        self.colors = colors
        self.actions = actions

        # Shared convolutional layers
        self.shared_layers = nn.Sequential(
            nn.Conv2d(colors, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(512, actions)

        # Value head
        self.value_head = nn.Linear(512, 1)

        if model_path is not None:
            self.load_state_dict(torch.load(model_path, weights_only=True))
            parts = model_path.split("/")
            run_id = parts[1]
            model_part = parts[2].split("_")[1].split(".")[0]
            self.model_name = f"{run_id}-{model_part}"
            print(f"Loaded model: {self.model_name}")

    def forward(self, x):
        if x.shape[1] == x.shape[2]:
            # Change to (batch, channel, width, height)
            x = x.permute(0, 3, 1, 2)
        
        x = x / 255.0
        features = self.shared_layers(x)
        action_logits = self.policy_head(features)
        state_values = self.value_head(features)
        return action_logits, state_values

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def load_actor_critic(model_path="runs/ppo_model.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actor_critic = ActorCriticNetwork(model_path=model_path).to(device).eval()
        return actor_critic, device