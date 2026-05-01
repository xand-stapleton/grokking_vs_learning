import copy
import random
from timeit import default_timer as timer

import numpy as np
import torch
from tqdm.auto import tqdm

from helper_functions.data_tools import sample_indices
from helper_functions.train_tools import get_samples
from models.bases import AnalysableModel


class Model(AnalysableModel):
    def __init__(
        self,
        hp,
        optimiser,
        network,
        train_dataloader,
        test_dataloader,
        fim_dataloader=None,
        save_name="",
    ):
        # Initialise the parent class.
        super().__init__()

        # Instantiate the optimiser
        self.optimiser = optimiser

        # Load in the hyperparameters config
        self.hp = hp

        # Instantiate the network
        self.network = network

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.fim_dataloader = fim_dataloader

        self.save_name = save_name

        # Random seed setting stuff
        random.seed(self.hp.sgd_seed)
        for set_seed in [
            torch.manual_seed,
            torch.cuda.manual_seed_all,
            np.random.seed,
        ]:
            set_seed(self.hp.sgd_seed)

    def train(
        self,
        epochs,
        save_interval,
        train_loader,
        test_loader,
        one_run_object,
        loss_criterion,
        device,
        evaluate_dic,
        fisher_loader=None,
    ):
        start_time = timer()
        first_time_training = True
        epochs = epochs  # how many runs through entire training data
        save_models = True
        save_interval = save_interval
        fix_norm = False
        compare_models = 1  # m in slack discussion
        weight_samples = []
        bias_samples = []
        random_weight_samples = 1000
        random_bias_sample = 5
        optimiser = self.optimiser(
            params=self.network.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
        )
        self.network.to(device)

        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)

        with torch.no_grad():
            weight_indices = [
                sample_indices(weight, random_weight_samples)
                for weight in self.network.parameters()
                if weight.dim() > 1
            ]
            bias_indices = [
                sample_indices(bias, random_bias_sample)
                for bias in self.network.parameters()
                if bias.dim() == 1
            ]

        inital_weights_sample, initial_biases_sample = get_samples(
            self.network, weight_indices, bias_indices
        )
        weight_samples.append(inital_weights_sample)
        bias_samples.append(initial_biases_sample)

        if save_models:
            save_dict0 = {"model": self.network.state_dict()}
            one_run_object.models[0] = save_dict0

        if first_time_training == True:
            train_losses = []
            test_losses = []
            train_accuracy = []
            test_accuracy = []
            norms = []
            decile_split = []
            back_compare = []
            linear_decomposition_dict = {
                "full_loss_test": [],
                "linear_loss_test": [],
                "non_linear_loss_test": [],
                "linear_norm_test": [],
                "non_linear_norm_test": [],
                "diff_norm_test": [],
                "full_loss_train": [],
                "linear_loss_train": [],
                "non_linear_loss_train": [],
                "linear_norm_train": [],
                "non_linear_norm_train": [],
                "diff_norm_train": [],
            }
        else:
            print("Starting additional training")
            epochs = 2900
        if fix_norm:
            norm = np.sqrt(
                sum(
                    param.pow(2).sum().item()
                    for param in self.network.parameters()
                )
            )

        # start training loop.
        for i in tqdm(range(epochs)):
            # Flatten input tensors to two index object with shape (batch_size, input_dims) using .view()
            train_loss = 0.0
            train_acc = 0.0

            # Log the evaluation quantities before the training step so that you get the initial random epoch.
            modweights = ["model.0.weight", "model.2.weight"]
            self.evaluate_analysis_quantities(
                evaluate_dic,
                back_compare,
                compare_models,
                i,
                linear_decomposition_dict,
                train_loader,
                test_loader,
                loss_criterion,
                modweights,
                fisher_loader=fisher_loader,
            )
            epoch_weight_samples, epoch_bias_samples = get_samples(
                self.network, weight_indices, bias_indices
            )
            weight_samples.append(epoch_weight_samples)
            bias_samples.append(epoch_bias_samples)

            for batch in train_loader:
                X_train, y_train = batch
                optimiser.zero_grad()

                y_pred = self.network(X_train).to(device)
                y_train = y_train.float().to(device)
                loss = loss_criterion(y_pred, y_train.to(device))
                loss.backward()

                optimiser.step()
                train_loss += loss.item()

                train_acc += (
                    (y_pred.argmax(dim=1) == y_train.argmax(dim=1)).sum().item()
                )

                if fix_norm:
                    with torch.no_grad():
                        new_norm = np.sqrt(
                            sum(
                                param.pow(2).sum().item()
                                for param in self.network.parameters()
                            )
                        )
                        for param in self.network.parameters():
                            param.data *= norm / new_norm

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            train_acc /= train_size  # train loss default is to average over a batch so you only need to divide by the number of batches to get the average loss.
            train_accuracy.append(train_acc)

            # Log the quantities we care about

            # Update overall train loss (most recent batch) & accuracy (of all batches) for the epoch

            test_loss, test_acc = self.test(
                test_loader, loss_criterion, test_size, device
            )
            test_losses.append(test_loss)
            test_accuracy.append(test_acc)

            # Update test loss & accuracy for the epoch
            if save_models and i % save_interval == 0:
                save_dict = {
                    "model": copy.deepcopy(self.network.state_dict()),
                    "optimiser": copy.deepcopy(optimiser.state_dict()),
                    # 'scheduler': scheduler.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "epoch": i,
                }
                one_run_object.models[i] = save_dict

        print(
            f"\nDuration: {timer() - start_time:.0f} seconds"
        )  # print the time elapsed

        one_run_object.train_losses = train_losses
        one_run_object.test_losses = test_losses
        one_run_object.train_accuracies = train_accuracy
        one_run_object.test_accuracies = test_accuracy

        # Analysable quantities
        one_run_object.linear_decomposition = linear_decomposition_dict

        for key in self.analysable_quantites.keys():
            if key == "iprs":
                one_run_object.iprs = self.analysable_quantites[key]
            if key == "norms":
                one_run_object.norms = norms
            if key == "euclidcosine":
                modelcosines = self.analysable_quantites[key]
                stepscosines = self.analysable_quantites["euclidcosinesteps"]
                if len(modelcosines) > 0:
                    modelcosines = torch.stack(modelcosines).numpy()
                if len(stepscosines) > 0:
                    stepscosines = torch.stack(stepscosines).numpy()
                one_run_object.euclidcosine = modelcosines
                one_run_object.euclidcosinesteps = stepscosines
            if key == "decile_steps":
                if len(decile_split) > 0:
                    decile_splits = torch.stack(decile_split).numpy()
                else:
                    decile_splits = []
                    one_run_object.decile_steps = decile_splits

        return one_run_object

    def test(self, test_loader, loss_criterion, test_size, device):
        # Run the testing batches
        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0
            for batch in test_loader:
                X_test, y_test = batch
                y_val = self.network(X_test).to(device)
                float_y = y_test.float().clone().to(device)
                loss = loss_criterion(y_val, float_y.to(device))
                test_loss += loss.item()
                if loss_criterion == "MSE":
                    test_acc += (
                        (y_val.argmax(dim=1) == float_y.argmax(dim=1))
                        .sum()
                        .item()
                    )
                else:
                    test_acc += (
                        (y_val.argmax(dim=1) == float_y.argmax(dim=1))
                        .sum()
                        .item()
                    )

                test_loss /= len(test_loader)
                test_acc /= test_size

            return test_loss, test_acc
