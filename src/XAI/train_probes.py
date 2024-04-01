import time
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd

from src.DRL.qnetwork import QNetwork
from src.XAI.concept import Concept
from src.XAI.concept_probes import train_probes
from src.XAI.concepts import concept_instances


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
    plt.savefig(f'{concept.folder_path}probe_training.png')
    
def get_hyperparams_combinations(hyperparams_dict):
    keys, value_tuples = zip(*hyperparams_dict.items())
    all_combinations = product(*value_tuples)
    
    # Create a list of dictionaries, each representing a unique combination of hyperparameters
    hyperparams_combinations = [
        dict(zip(keys, combination)) for combination in all_combinations
    ]
    return hyperparams_combinations


if __name__ == '__main__':
    max_size = 10000
    layer = 4
    
    # Use (x, ) for single values
    hyperparam_ranges = {
        'lr': (0.01, 0.001, 0.0001, 0.00001),
        'batch_size': (64,),
        'lambda_l1': (0.0,),
        'patience': (5,),
        'epochs': (100,)
    }
    hyperparams = get_hyperparams_combinations(hyperparam_ranges)
    print(f'Number of hyperparameters configs: {len(hyperparams)}')

    #model_path = QNetwork.find_newest_model()
    model_path = "runs/20240317-112025/model_10000000.pt"
    model = QNetwork(model_path=model_path)
    
    env_steps = Concept.load_concept_data()
    for concept in concept_instances.values():
        start_time = time.time()
        concept.prepare_data(env_steps, max_size=max_size)
        concept.summary()
        
        layer_probes, layer_info, best_hyperparams = train_probes(model, concept, hyperparams, [layer])
        concept.save_torch_probe(layer_probes[layer], model.model_name, layer, layer_info[layer]['test_score'][-1])
        save_plot(concept, layer_info[layer])
        print(f'Layer {layer} | Test score: {layer_info[layer]["test_score"][-1]:.3f} | Best hyperparams: {best_hyperparams} | Time: {time.time() - start_time:.2f}s\n')