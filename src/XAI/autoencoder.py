import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, latent_variables):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, latent_variables)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_variables):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_variables, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 8, stride=4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.permute(0, 2, 3, 1)
        x = x * 255.0
        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim, train_obs, test_obs, device):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.train_obs = train_obs
        self.test_obs = test_obs
        self.device = device
        self.to(device)
        
        self.criterion = nn.MSELoss()
        #self.criterion = nn.L1Loss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_model(self, num_epochs, batch_size, learning_rate):
        self.train()  # Set the model to training mode

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(0, len(self.train_obs), batch_size):
                batch = self.train_obs[i:i + batch_size].to(self.device)
                optimizer.zero_grad()
                outputs = self(batch)
                loss = self.criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            average_loss = total_loss / len(self.train_obs)
            #if epoch % 10 == 0:
            val_loss = self.validate(batch_size)
            print(f'Epoch {epoch}/{num_epochs} | Train Loss: {average_loss} | Val Loss: {val_loss}')
            scheduler.step(val_loss)

    def validate(self, batch_size):
        self.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(self.test_obs), batch_size):
                batch = self.test_obs[i:i + batch_size].to(self.device)
                outputs = self(batch)
                loss = self.criterion(outputs, batch)
                total_val_loss += loss.item()
        self.train()  # Set back to train mode
        return total_val_loss / len(self.test_obs)
            