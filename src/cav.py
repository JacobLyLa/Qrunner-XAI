import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from PIL import Image
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from concepts import concept_instances
from train_model import load_model
from utils import prepare_folders

class CAV:
    def __init__(self, concept, random_concept, q_network, activation_layers, device):
        self.concept = concept
        self.random_concept = random_concept
        self.q_network = q_network
        self.activation_layers = activation_layers
        self.device = device

        self.activation_outputs = []
        self.cavs = {layer: None for layer in activation_layers}
        self.intercepts = {layer: None for layer in activation_layers}
        self.coeficients = {layer: None for layer in activation_layers}
        self.lm_test_set_accuracies = {layer: None for layer in activation_layers}
        self.xgb_test_set_accuracies = {layer: None for layer in activation_layers}

        # prepare concept presence and absence observations
        self.concept_presence_obs = concept.get_presence_observations()
        self.concept_absence_obs = concept.get_absence_observations()
        # cut off to make a balanced dataset
        self.sample_size = min(len(self.concept_presence_obs), len(self.concept_absence_obs))
        self.concept_presence_obs = self.concept_presence_obs[:self.sample_size]
        self.concept_absence_obs = self.concept_absence_obs[:self.sample_size]
        # obs to tensors
        self.concept_presence_obs = torch.Tensor(self.concept_presence_obs).to(self.device)
        self.concept_absence_obs = torch.Tensor(self.concept_absence_obs).to(self.device)
        # prepare random observations and images (for visualizing)
        self.random_observations = random_concept.get_presence_observations()
        self.random_observations = torch.Tensor(self.random_observations).to(self.device)
        self.random_images = random_concept.get_presence_images()

        self.hooks = []
        self._register_hooks()

    def _register_hooks(self): 
        # TODO: not optimal way to do it, since each concept will add new hooks...
        # use .close() after use...
        def hook_fn(module, input, output):
            if len(output.shape) > 1:
                # flatten all but first dimension (so it works on conv layers)
                output = output.flatten(start_dim=1)
            self.activation_outputs.append(output)
        for layer in self.activation_layers:
            hook = self.q_network.network[layer].register_forward_hook(hook_fn)
            self.hooks.append(hook)

    def _generate_cav(self, layer, k=10, test_xgb=False, verbose=False):
        layer_index = self.activation_layers.index(layer)
        concept_activations_presence = self.activation_outputs[layer_index].cpu().numpy()
        concept_activations_absence = self.activation_outputs[len(self.activation_layers) + layer_index].cpu().numpy()
        # Create labels for the classes
        y_class0 = np.zeros(self.sample_size)
        y_class1 = np.ones(self.sample_size)

        # Combine data points and labels
        x = np.concatenate((concept_activations_presence, concept_activations_absence))
        y = np.concatenate((y_class0, y_class1))

        # Create a KFold cross-validator
        fold_accuracies = []
        kf = KFold(n_splits=k, shuffle=True)

        # Iterate through the folds
        for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit linear model on the training data for this fold
            lm = linear_model.SGDClassifier(loss="log_loss")
            lm.fit(x_train, y_train)
            
            # Calculate accuracy on the test set for this fold
            test_set_accuracy = accuracy_score(y_test, lm.predict(x_test))
            
            # Store the accuracy for this fold
            fold_accuracies.append(test_set_accuracy)

        # Calculate the average accuracy across all folds
        final_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        if verbose:
            print(f"Average Linear test accuracy across {len(fold_accuracies)} folds: {final_accuracy:.2f}")
        self.lm_test_set_accuracies[layer] = final_accuracy

        # Fit the final model with all the data
        final_lm = linear_model.SGDClassifier(loss="log_loss")
        final_lm.fit(x, y)

        if test_xgb:
            # TODO: test xgb
            pass
            '''
            fold_accuracies = []
            kf = KFold(n_splits=10, shuffle=True)

            # Iterate through the folds
            for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Calculate accuracy on the test set for this fold
                test_set_accuracy = accuracy_score(y_test, xgb_model.predict(x_test))
                
                # Store the accuracy for this fold
                fold_accuracies.append(test_set_accuracy)

            # Calculate the average accuracy across all folds
            final_accuracy = sum(fold_accuracies) / len(fold_accuracies)

            if verbose:
                print(f"[{layer}] XGB test accuracy: {final_accuracy:.2f}")
            self.xgb_test_set_accuracies[layer] = final_accuracy
            '''

        # Save the CAV
        coefficients = final_lm.coef_[0]
        norm = np.linalg.norm(coefficients)
        normalized_coefficients = coefficients / norm
        self.cavs[layer] = -normalized_coefficients
        self.intercepts[layer] = final_lm.intercept_[0]
        self.coeficients[layer] = coefficients

    def generate_cavs(self, k=10, test_xgb=False, save=False, verbose=False):
        with torch.no_grad():
            self.q_network(self.concept_presence_obs)
            self.q_network(self.concept_absence_obs)
        if verbose:
            print(f"Generating CAVs for concept {self.concept.name}...")
        for layer in self.activation_layers:
            self._generate_cav(layer, k=k, test_xgb=test_xgb, verbose=verbose)
        # save cavs to disk
        if save:
            cavs = [self.cavs[layer] for layer in self.activation_layers]
            self.concept.save_cavs(cavs, self.activation_layers)
        # clear activation outputs
        self.activation_outputs = []

    def sort_random_images(self):
        # choose CAV with highest test set accuracy
        best_layer = max(self.lm_test_set_accuracies, key=self.lm_test_set_accuracies.get)
        best_layer_index = self.activation_layers.index(best_layer)
        print(f"best layer: {best_layer}")
        cav = self.cavs[best_layer]
        # forward pass random observations through the model
        with torch.no_grad():
            self.q_network(self.random_observations)
        # Get the activations captured by the hook
        random_activations = self.activation_outputs[best_layer_index].cpu().numpy()
        # Find the distance from the decision boundary to origin
        distance_to_origin = abs(self.intercepts[best_layer]) / np.linalg.norm(self.coeficients[best_layer])
        origin_vector = cav * distance_to_origin
        random_activations = random_activations + origin_vector
        # Compare each random activation with CAV using cosine similarity
        cosine_similarities = np.dot(random_activations, cav)/(np.linalg.norm(random_activations, axis=1) * np.linalg.norm(cav))
        # Sort random_images by cosine_similarities
        random_images_similarity = sorted(zip(self.random_images, cosine_similarities), key=lambda x: x[1], reverse=True)
        # Save sorted random_images
        prepare_folders(f"{self.concept.path}/random_sorted_images")
        for i, image_similarity in enumerate(random_images_similarity):
            image, similarity = image_similarity
            image = Image.fromarray(image)
            image.save(f"{self.concept.path}/random_sorted_images/image_{i}.png")

    def save_accuracies(self):
        plt.clf()
        x = np.arange(len(self.activation_layers))
        y1 = np.array(list(self.lm_test_set_accuracies.values()))
        if None not in self.xgb_test_set_accuracies.values():
            y2 = np.array(list(self.xgb_test_set_accuracies.values()))
            plt.plot(x, y2, marker="o", linestyle="dashed", label="XGB Model")


        plt.plot(x, y1, marker="o", linestyle="dashed", label="Linear Model")
        plt.xticks(x, self.activation_layers)
        plt.ylim(0.0, 1.1)
        plt.title(f"Accuracy vs Layer for concept: {self.concept.name}")
        plt.xlabel("DQN layer")
        plt.ylabel("Test accuracy")
        plt.legend()
        # save accuracy over layers plot
        plt.savefig(f"{self.concept.path}/accuracies.png")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def close(self):
        self.remove_hooks()
        self.concept_presence_obs.detach()
        self.concept_absence_obs.detach()
        self.random_observations.detach()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = load_model("../runs/20230917-010123/models/model_3162277.pt").to(device)
    for i, layer in enumerate(q_network.network):
        print(f"Layer {i}: {layer}")
    random_concept = concept_instances[0]
    activation_layers = [1,2,3,4,5,6,8,9]
    for concept in concept_instances:
        cav = CAV(concept, random_concept, q_network, activation_layers, device)
        cav.generate_cavs(test_xgb=True, save=True, verbose=True)
        cav.sort_random_images()
        cav.save_accuracies()
        cav.close()