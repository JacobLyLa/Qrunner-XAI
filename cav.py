import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from PIL import Image
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from concepts import concept_instances
from dqn import load_model
from utils import prepare_folders

# TODO: make CAV normalized and dont reshape it?

class CAV:
    def __init__(self, concept, random_concept, q_network, activation_layers, device):
        self.concept = concept
        self.random_concept = random_concept
        self.q_network = q_network
        self.activation_layers = activation_layers
        self.device = device

        self.activation_outputs = []
        self.cavs = {layer: None for layer in activation_layers}
        self.linear_models = {layer: None for layer in activation_layers}
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

    def _register_hooks(self): # TODO: not optimal way to do it, since each concept will add new hooks...
        def hook_fn(module, input, output):
            if len(output.shape) > 1:
                # flatten all but first dimension (so it works on conv layers)
                output = output.flatten(start_dim=1)
            self.activation_outputs.append(output)
        for layer in self.activation_layers:
            hook = self.q_network.network[layer].register_forward_hook(hook_fn)
            self.hooks.append(hook)
            # self.q_network.network[layer].register_forward_hook(hook_fn)

    def _generate_cav(self, layer):
        layer_index = self.activation_layers.index(layer)
        concept_activations_presence = self.activation_outputs[layer_index].cpu().numpy()
        concept_activations_absence = self.activation_outputs[len(self.activation_layers) + layer_index].cpu().numpy()
        # Create labels for the classes
        y_class0 = np.zeros(self.sample_size)
        y_class1 = np.ones(self.sample_size)

        # Combine data points and labels
        x = np.concatenate((concept_activations_presence, concept_activations_absence))
        y = np.concatenate((y_class0, y_class1))

        # Split into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

        # Fit linear model on the data
        lm = linear_model.SGDClassifier()
        lm.fit(x_train, y_train)
        test_set_accuracy = accuracy_score(y_test, lm.predict(x_test))
        print(f"[{layer}] Linear test accuracy:", test_set_accuracy)
        self.lm_test_set_accuracies[layer] = test_set_accuracy
        self.linear_models[layer] = lm

        # Fit XGBoost model on the data
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
        param['eval_metric'] = 'auc'
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 5
        bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=evallist, verbose_eval=False)
        preds = bst.predict(dtest)
        test_set_accuracy = accuracy_score(y_test, preds.round())
        print(f"[{layer}] XGB test accuracy:", test_set_accuracy)
        self.xgb_test_set_accuracies[layer] = test_set_accuracy

        # Save the CAV
        self.cavs[layer] = -lm.coef_[0]

    def generate_cavs(self):
        with torch.no_grad():
            self.q_network(self.concept_presence_obs)
            self.q_network(self.concept_absence_obs)
        print(f"Generating CAVs for concept {self.concept.name}...")
        for layer in self.activation_layers:
            self._generate_cav(layer)
        # save cavs to disk
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
        # Compare each random activation with CAV using cosine similarity
        cosine_similarities = np.dot(random_activations, cav)/(np.linalg.norm(random_activations, axis=1) * np.linalg.norm(cav))
        # Sort random_images by cosine_similarities
        random_images_similarity = sorted(zip(self.random_images, cosine_similarities), key=lambda x: x[1], reverse=True)
        # Save sorted random_images
        prepare_folders(f"concepts/{self.concept.name}/random_sorted_images")
        for i, image_similarity in enumerate(random_images_similarity):
            image, similarity = image_similarity
            image = Image.fromarray(image)
            image.save(f"concepts/{self.concept.name}/random_sorted_images/image_{i}.png")

    def plot_accuracies(self):
        # plot accuracies
        x = np.arange(len(self.activation_layers))
        y = np.array(list(self.lm_test_set_accuracies.values()))
        plt.plot(x, y, marker="o", linestyle="dashed", label=self.concept.name)
        plt.xticks(x, self.activation_layers)
        plt.ylim(0.0, 1.1)
        plt.title(f"Accuracy vs Layer")
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.legend()
        # plt.show()

    def save_accuracies(self):
        x = np.arange(len(self.activation_layers))
        y1 = np.array(list(self.lm_test_set_accuracies.values()))
        y2 = np.array(list(self.xgb_test_set_accuracies.values()))

        plt.clf()

        plt.plot(x, y1, marker="o", linestyle="dashed", label="Linear")
        plt.plot(x, y2, marker="o", linestyle="dashed", label="XGB")
        plt.xticks(x, self.activation_layers)
        plt.ylim(0.0, 1.1)
        plt.title(f"Accuracy vs Layer for concept: {self.concept.name}")
        plt.xlabel("DQN layer")
        plt.ylabel("Test accuracy")
        plt.legend()
        # save in cavs folder
        plt.savefig(f"concepts/{self.concept.name}/cavs/accuracies.png")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_network = load_model("runs/dqn_breakout/20230901-100931/models/model_5500000.pt").to(device)
# print network architecture
for i, layer in enumerate(q_network.network):
    print(f"Layer {i}: {layer}")
random_concept = concept_instances[0]
# activation_layers = [1,3,5,8]
activation_layers = list(range(1, 9))
for concept in concept_instances:
    cav = CAV(concept, random_concept, q_network, activation_layers, device)
    cav.generate_cavs()
    #cav.sort_random_images()
    cav.save_accuracies()
    #cav.plot_accuracies()
    cav.remove_hooks()
plt.show()