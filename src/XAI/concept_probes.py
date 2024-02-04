import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score as r2_score_torch

warnings.filterwarnings('ignore', category=ConvergenceWarning)

def binary_score(preds, y):
    preds = preds > 0.5
    correct = (preds == y).float()
    acc = correct.sum() / len(correct)
    return 2*acc-1

def validate_probe(preds, y, binary):
    score = binary_score(preds, y) if binary else r2_score_torch(preds, y)
    return score.cpu().item()

def create_data(model, concept):
    train_obs = torch.tensor(concept.train_obs.copy()).float()
    test_obs = torch.tensor(concept.test_obs.copy()).float()
    input_max = train_obs.max()
    _, acts_dict_train = model(train_obs, return_acts=True)
    _, acts_dict_test = model(test_obs, return_acts=True)
    acts_dict_train_test = {}
    # Include original observation data
    acts_dict_train_test[-1] = {'train_acts': train_obs/input_max, 'test_acts': test_obs/input_max}
    for k in acts_dict_train.keys():
        acts_dict_train_test[k] = {'train_acts': acts_dict_train[k], 'test_acts': acts_dict_test[k]}

    # Add concept values
    train_values = torch.tensor(concept.train_values, dtype=torch.float32)
    test_values = torch.tensor(concept.test_values, dtype=torch.float32)

    return acts_dict_train_test, train_values, test_values


def _train_probe(binary, hyperparams, train_acts, test_acts, train_values, test_values):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_eval = []
    test_eval = []
    test_score = []
    # Scale activations by dividing by the maximum abs activation value
    #train_max_act = torch.max(torch.abs(train_acts))
    #train_acts = train_acts / train_max_act
    #test_acts = test_acts / train_max_act
    #print(f'Train max activation: {train_max_act}')
    
    # Create data loaders
    train_dataset = TensorDataset(train_acts.to(device), train_values.to(device))
    test_dataset = TensorDataset(test_acts.to(device), test_values.to(device))
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    # Create probe
    num_activation_inputs = np.prod(train_loader.dataset[0][0].shape)
    if binary:
        probe = nn.Sequential(nn.Flatten(), nn.Linear(num_activation_inputs, 1), nn.Sigmoid())
    else:
        probe = nn.Sequential(nn.Flatten(), nn.Linear(num_activation_inputs, 1))
    probe = probe.to(device)
    optimizer = optim.Adam(probe.parameters(), lr=hyperparams['lr'])
    loss_fn = nn.MSELoss()
    
    # Initialize variables for early stopping
    best_test_score = -float('inf')
    epochs_without_improvement = 0

    for epoch in range(hyperparams['epochs']):
        probe.train()
        total_train_loss = 0.0
        for batch_obs, batch_values in train_loader:
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
                outputs = probe(batch_obs)
                loss = loss_fn(outputs.squeeze(1), batch_values)
                
                total_test_loss += loss.item()
                
                eval_score = validate_probe(outputs.squeeze(1), batch_values, binary)
                if eval_score > 1 or np.isnan(eval_score):
                    assert False, f'Invalid score: {eval_score}'
                scores.append(eval_score)
        score = np.mean(scores)

        train_eval.append(total_train_loss/len(train_loader))
        test_eval.append(total_test_loss/len(test_loader))
        test_score.append(score)
        # Early stopping
        if score > best_test_score:
            best_test_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hyperparams['patience']:
                break
    probe = probe.cpu()
    return probe, {'train_eval': train_eval, 'test_eval': test_eval, 'test_score': test_score}

def train_probes(model, concept, hyperparams, layers):
    hyperparam_tuning = False
    if isinstance(hyperparams, list):
        hyperparam_tuning = True
        
    acts_dict_train_test, train_values, test_values = create_data(model, concept)
    layer_probes = {}
    layer_info = {}
    for layer in layers:
        train_acts = acts_dict_train_test[layer]['train_acts']
        test_acts = acts_dict_train_test[layer]['test_acts']
        if hyperparam_tuning:
            best_hyperparams = []
            best_probe = None
            best_hyperparam = None
            best_info = None
            best_score = -float('inf')
            for hyperparam_instance in hyperparams:
                probe, info = _train_probe(concept.binary, hyperparam_instance, train_acts, test_acts, train_values, test_values)
                if info['test_score'][-1] > best_score:
                    best_probe = probe
                    best_hyperparam = hyperparam_instance
                    best_info = info
                    best_score = info['test_score'][-1]
            probe = best_probe
            info = best_info
            best_hyperparams.append(best_hyperparam)
        else:
            probe, info = _train_probe(concept.binary, hyperparams, train_acts, test_acts, train_values, test_values)
        layer_probes[layer] = probe
        layer_info[layer] = info
    if hyperparam_tuning:
        return layer_probes, layer_info, best_hyperparams
    return layer_probes, layer_info