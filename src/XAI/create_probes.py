import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score as r2_score_torch

from src.DRL.qnetwork import QNetwork
from src.XAI.concept import Concept
from src.XAI.concepts import concept_instances
from src.XAI.concept_probes import train_probes


def save_plot(concept, info):
    pd_info = pd.DataFrame(info)
    pd_info = pd_info.reset_index().rename(columns={'index': 'epoch'})
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Plot train and test loss
    axs[0].plot('epoch', 'train_eval', data=pd_info, label='Train Loss')
    axs[0].plot('epoch', 'test_eval', data=pd_info, label='Test Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Train and Test Loss')

    # Plot test score
    axs[1].plot('epoch', 'test_score', data=pd_info, label='Test Score', color='green')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Score')
    axs[1].set_ylim(0, 1)
    axs[1].legend()
    axs[1].set_title('Test Score')

    plt.tight_layout()
    plt.savefig(f'{concept.folder_path}training_info.png')
    
def get_hyperparams_list(lr, batch_size, lambda_l1, patience, epochs):
    hyperparams = []
    for lr_ in lr:
        for batch_size_ in batch_size:
            hyperparams.append({
                'lr': lr_,
                'lambda_l1': lambda_l1,
                'patience': patience,
                'epochs': epochs,
                'batch_size': batch_size_,
            })
    return hyperparams

# TODO: rename to train_probes
if __name__ == '__main__':
    lr = (0.1, 0.01, 0.001)
    batch_size = (64, 256, 512)
    step_size = 5000
    layer = 4

    model_path = QNetwork.find_newest_model()
    model = QNetwork(frame_stacks=4, model_path=model_path)
    print(f"Using model: {model_path}")
    
    hyperparams = get_hyperparams_list(lr, batch_size, lambda_l1=0.0, patience=10, epochs=200)
    print(f'Number of hyperparameters configs: {len(hyperparams)}')
    
    env_steps = Concept.load_concept_data()
    for concept in concept_instances.values():
        concept.prepare_data(env_steps, max_size=step_size)
        concept.summary()
        
        layer_probes, layer_info, best_hyperparams = train_probes(model, concept, hyperparams, [layer])
        print(f'Layer {layer} | Test score: {layer_info[layer]["test_score"][-1]:.3f} | Best hyperparams: {best_hyperparams}')
        save_plot(concept, layer_info[layer])
        concept.save_torch_probe(layer_probes[layer], model.model_name, layer, layer_info[layer]['test_score'][-1])