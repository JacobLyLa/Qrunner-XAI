import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score


def binary_score(preds, y):
    preds = preds > 0.5
    correct = (preds == y).float()
    acc = correct.sum() / len(correct)
    return 2*acc-1

def validate_probe(preds, y, binary):
    score = binary_score(preds, y) if binary else r2_score(preds, y)
    return score.cpu().item()

def create_data(game_data, model, concept):
    concept.prepare_data(game_data)
    train_obs = torch.tensor(concept.train_obs.copy()).float()
    test_obs = torch.tensor(concept.test_obs.copy()).float()
    t_max = train_obs.max()
    _, acts_dict_train = model(train_obs, return_acts=True)
    _, acts_dict_test = model(test_obs, return_acts=True)
    acts_dict_train_test = {}
    for k in acts_dict_train.keys():
        acts_dict_train_test[k] = {'train_acts': acts_dict_train[k], 'test_acts': acts_dict_test[k]}
    # Include original observation data
    acts_dict_train_test['obs'] = {'train_acts': train_obs/t_max, 'test_acts': test_obs/t_max}

    # Add concept values
    train_values = torch.tensor(concept.train_values, dtype=torch.float32)
    test_values = torch.tensor(concept.test_values, dtype=torch.float32)

    return acts_dict_train_test, train_values, test_values


def _train_probe(binary, hyperparams, train_acts, test_acts, train_values, test_values):
    # create data loaders
    train_dataset = TensorDataset(train_acts, train_values)
    test_dataset = TensorDataset(test_acts, test_values)
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    # create probe
    num_activation_inputs = np.prod(train_loader.dataset[0][0].shape)
    if binary:
        probe = nn.Sequential(nn.Flatten(), nn.Linear(num_activation_inputs, 1), nn.Sigmoid())
    else:
        probe = nn.Sequential(nn.Flatten(), nn.Linear(num_activation_inputs, 1))
    #probe.to(device)
    optimizer = optim.Adam(probe.parameters(), lr=hyperparams['lr'])
    loss_fn = nn.MSELoss()

    # Initialize variables for early stopping
    best_test_score = -float('inf')
    best_probe = None  # Variable to store the best probe
    epochs_without_improvement = 0

    log = ""

    for epoch in range(hyperparams['epochs']):
        probe.train()
        total_train_loss = 0.0
        for batch_obs, batch_values in train_loader:
            #batch_obs = batch_obs.to(device)
            #batch_values = batch_values.to(device)

            optimizer.zero_grad()
            outputs = probe(batch_obs)
            loss = loss_fn(outputs.squeeze(1), batch_values)
            
            # Compute the L1 penalty
            l1_penalty = sum(p.abs().sum() for p in probe.parameters())
            total_loss = loss + hyperparams['lambda_l1'] * l1_penalty
            
            total_train_loss += total_loss.item()
            
            total_loss.backward()
            optimizer.step()

        # Validation phase
        probe.eval()
        total_test_loss = 0.0
        scores = []

        with torch.no_grad():
            for batch_obs, batch_values in test_loader:
                #batch_obs = batch_obs.to(device)
                #batch_values = batch_values.to(device)

                outputs = probe(batch_obs)
                loss = loss_fn(outputs.squeeze(1), batch_values)
                
                total_test_loss += loss.item()
                
                scores.append(validate_probe(outputs.squeeze(1), batch_values, binary))
        score = np.mean(scores)

        log += f"Epoch {epoch+1}/{hyperparams['epochs']} - Train loss: {total_train_loss/len(train_loader):.4f} - Test loss: {total_test_loss/len(test_loader):.4f} - Test score: {score:.4f}\n"

        # Early stopping
        if score > best_test_score:
            best_test_score = score
            epochs_without_improvement = 0
            # Save a copy of the best probe
            best_probe = copy.deepcopy(probe)

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hyperparams['patience']:
                break

    #if score < 0.3:
    #    print(log)

    return best_probe, best_test_score  # Return the best probe and the best score


def train_probe(game_data, model, concept, layer, hyperparams):
    acts_dict_train_test, train_values, test_values = create_data(game_data, model, concept)
    train_acts = acts_dict_train_test[layer]['train_acts']
    test_acts = acts_dict_train_test[layer]['test_acts']
    probe, score = _train_probe(concept.binary, hyperparams, train_acts, test_acts, train_values, test_values)
    #print(f"Layer {layer} - Score: {score:.4f}")
    return probe, score

def train_probes(game_data, model, concept, hyperparams):
    acts_dict_train_test, train_values, test_values = create_data(game_data, model, concept) # Reuse for same concept but different layers
    layer_scores = {}
    for layer in acts_dict_train_test.keys():
        train_acts = acts_dict_train_test[layer]['train_acts']
        test_acts = acts_dict_train_test[layer]['test_acts']
        probe, score = _train_probe(concept.binary, hyperparams, train_acts, test_acts, train_values, test_values)
        layer_scores[layer] = score
        #print(f"Layer {layer} - Score: {score:.4f}")
    return layer_scores