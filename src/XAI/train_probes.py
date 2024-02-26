import time

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
    
def get_hyperparams_list(lr, batch_size, lambda_l1, patience, epochs):
    hyperparams = []
    for lr_ in lr:
        for batch_size_ in batch_size:
            for lambda_l1_ in lambda_l1:
                hyperparams.append({
                    'lr': lr_,
                    'lambda_l1': lambda_l1_,
                    'patience': patience,
                    'epochs': epochs,
                    'batch_size': batch_size_,
                })
    return hyperparams

if __name__ == '__main__':
    max_size = 20000
    layer = 4
    frame_stacks = 1
    
    patience = 10
    epochs = 200
    lr = (0.1, 0.01, 0.001)
    batch_size = (256, 512)
    lambdas = (0.0, 0.001)

    model_path = QNetwork.find_newest_model()
    model = QNetwork(frame_stacks=frame_stacks, model_path=model_path)
    print(f"Using model: {model_path}")
    
    hyperparams = get_hyperparams_list(lr, batch_size, lambda_l1=lambdas, patience=patience, epochs=epochs)
    print(f'Number of hyperparameters configs: {len(hyperparams)}')
    
    env_steps = Concept.load_concept_data()
    for concept in concept_instances.values():
        name = concept.name
        start_time = time.time()
        concept.prepare_data(env_steps, max_size=max_size)
        concept.summary()
        if len(concept.train_data) < 100:
            print(f'Not enough data')
            continue
        
        layer_probes, layer_info, best_hyperparams = train_probes(model, concept, hyperparams, [layer])
        print(f'Layer {layer} | Test score: {layer_info[layer]["test_score"][-1]:.3f} | Best hyperparams: {best_hyperparams} | Time: {time.time() - start_time:.2f}s')
        save_plot(concept, layer_info[layer])
        concept.save_torch_probe(layer_probes[layer], model.model_name, layer, layer_info[layer]['test_score'][-1])