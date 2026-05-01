import copy
from collections import defaultdict
from os import makedirs

import torch
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag

from helper_functions import data_tools, train_tools


class AnalysableModel:
    def __init__(self):
        # Treat loss criterion as property of training not model
        # Instantiate the optimiser
        self.optimiser = None

        # Load in the hyperparameters config
        self.hp = None

        # Keep track of previous network states to compare
        self.back_compare = None

        # Instantiate the network
        self.network = None

        # Instantiate the FIM dir
        self.fim_dir = None

        self.evaluate_dic = None
        self.analysable_quantites = {
            "iprs": [],
            "norms": [],
            "euclidcosine": [],
            "euclidcosinesteps": [],
            "decile_steps": [],
            "fims": [],
            # "linear_decomposition": {}, - this gets built in the training loop
        }

        self.length_elements = []

    # Change this function to include all the other optionally calculated quantities
    def evaluate_analysis_quantities(
        self,
        evaluate_dic,
        back_compare,
        compare_models,
        epoch,
        linear_decomposition_dict,
        train_dataloader,
        test_dataloader,
        loss_criterion,
        weight_keys=None,
        fisher_loader=None,
    ):
        for quantity, boolean in evaluate_dic.items():
            if quantity == "euclidcosine" and boolean:
                (
                    cos_sim_model,
                    cos_sim_steps,
                    back_steps_model,
                    front_steps_model,
                ) = data_tools.evaluate_cosines(
                    self.network,
                    back_compare,
                    epoch,
                    compare_models=compare_models,
                    weight_keys=weight_keys,
                )
                if cos_sim_model is not None and cos_sim_steps is not None:
                    # Needs to even if it returns None to update the back compare
                    self.analysable_quantites["euclidcosine"].append(
                        cos_sim_model
                    )
                    self.analysable_quantites["euclidcosinesteps"].append(
                        cos_sim_steps
                    )
                else:
                    # I may rue this later, but technically it's correct to
                    # leave blank for epochs we haven't calculated anything
                    # for.
                    self.analysable_quantites["euclidcosine"].append(
                        torch.full((len(weight_keys) + 1,), torch.nan)
                    )
                    self.analysable_quantites["euclidcosinesteps"].append(
                        torch.full((len(weight_keys) + 1,), torch.nan)
                    )

            if quantity == "iprs" and boolean:
                ipr2 = data_tools.calculate_ipr(self.network, 2)

                ipr4 = data_tools.calculate_ipr(self.network, 4)
                ipr05 = data_tools.calculate_ipr(self.network, 1 / 2)
                self.analysable_quantites[quantity].append([ipr2, ipr4, ipr05])

            if quantity == "norms" and boolean:
                self.analysable_quantites[quantity].append(
                    data_tools.calculate_weight_norm(self.network, 2)
                )

            if quantity == "linear_decomposition" and boolean:
                data_tools.evaluate_linear_decomposition(
                    linear_decomposition_dict,
                    self.network,
                    loss_criterion,
                    train_dataloader,
                    test_dataloader,
                    self.hp.device,
                )

            if quantity == "decile_steps" and boolean:
                split_into = 10
                if len(back_compare) > 1:
                    self.analysable_quantites[quantity].append(
                        data_tools.decile(
                            copy.deepcopy(back_compare[0]),
                            copy.deepcopy(back_compare[-1]),
                            split_into,
                        )
                    )
                else:
                    self.analysable_quantites[quantity].append(
                        torch.full((split_into,), torch.nan)
                    )

            if quantity == "fims" and boolean:
                if epoch % 100 == 0 or (epoch < 500):
                    # Make the FIM directory if it doesn't exist
                    self.fim_dir = self.save_name + "/fims/"
                    makedirs(self.fim_dir, exist_ok=True)
                    fim = FIM(
                        model=self.network,
                        loader=fisher_loader,
                        variant="regression",
                        representation=PMatDiag,
                        device=self.hp.device,
                    ).get_diag()

                    torch.save(fim, f"{self.fim_dir}/epoch-{epoch}.pt")
                    flattened_params = train_tools.flatten_parameters(
                        self.network
                    )
                    self.length_elements.append(
                        torch.sqrt(
                            sum(flattened_params * fim * flattened_params)
                        )
                        .detach()
                        .numpy()
                    )

    def fisher_prune(
        self,
        cutoff_beginning,
        cutoff_end,
        cutoff_num,
        fisher_loader,
        test_loader,
        loss_criterion,
        logspace=True,
        prune_by_cutoff=True,
        prune_layer=None,
    ):
        test_size = len(test_loader.dataset)
        # Calculate one final terminal FIM (in case the other one is old)
        # doubling up on effort
        if logspace:
            cutoffs = torch.logspace(cutoff_beginning, cutoff_end, cutoff_num)
        else:
            cutoffs = torch.linspace(cutoff_beginning, cutoff_end, cutoff_num)

        model_layers = list(self.network.modules())
        terminal_fim = FIM(
            model=self.network,
            loader=fisher_loader,
            variant="regression",
            representation=PMatDiag,
            device=self.hp.device,
        ).get_diag()

        # Check if self.original_network exists (tell us if the model has been
        # pruned before). If so, recover from the original network. Otherwise,
        # create a backup for next time
        if hasattr(self, "original_network") and self.original_network is not None:
            self.network = copy.deepcopy(self.original_network)
        else:
            self.original_network = copy.deepcopy(self.network)

        # If prune layer is None, prune all of the layers
        prune_layer_idx_beg = 0
        prune_layer_idx_end = len(terminal_fim)
        if prune_layer is not None:
            # Find the cumulative number of parameters per layer
            param_indices = [0]
            for name, param in self.network.named_parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    param_indices.append(param_indices[-1] + num_params)
            prune_layer_idx_beg = param_indices[prune_layer]
            prune_layer_idx_end = param_indices[prune_layer + 1]

        parameters_vec = torch.nn.utils.parameters_to_vector(
            self.network.parameters()
        )

        if prune_by_cutoff:
            # For each cutoff, prune the parameters
            cutoff_losses = []
            cutoff_accuracies = []
            if prune_layer is not None:
                for cutoff in cutoffs:
                    with torch.no_grad():
                        mask = (
                            terminal_fim < cutoff
                        )  # Identify elements below cutoff
                        mask[0:prune_layer_idx_beg] = (
                            False  # Exclude indices before the beginning of the range
                        )
                        mask[prune_layer_idx_end + 1 :] = (
                            False  # Exclude indices after the end of the range
                        )
                        parameters_vec[mask] = (
                            0  # Zero out only the selected elements
                        )

                    # Update the model with the new parameters
                    torch.nn.utils.vector_to_parameters(
                        parameters_vec, self.network.parameters()
                    )
                    test_loss, test_acc = self.test(
                        test_loader, loss_criterion, test_size, self.hp.device
                    )
                    cutoff_losses.append((cutoff, test_loss))
                    cutoff_accuracies.append((cutoff, test_acc))
            else:
                for cutoff in cutoffs:
                    with torch.no_grad():
                        parameters_vec[terminal_fim < cutoff] = 0
                    # Update the model with the new parameters
                    torch.nn.utils.vector_to_parameters(
                        parameters_vec, self.network.parameters()
                    )
                    test_loss, test_acc = self.test(
                        test_loader, loss_criterion, test_size, self.hp.device
                    )
                    cutoff_losses.append((cutoff, test_loss))
                    cutoff_accuracies.append((cutoff, test_acc))

            return cutoff_losses, cutoff_accuracies
        else:
            prune_losses = []
            prune_accuracies = []
            # Create an ordered list of the indices of the terminal fim
            sorting_order = torch.argsort(terminal_fim)
            for idx in sorting_order:
                if prune_layer_idx_beg <= idx <= prune_layer_idx_end:
                    with torch.no_grad():
                        parameters_vec[idx] = 0
                    # Update the model with the new parameters
                    torch.nn.utils.vector_to_parameters(
                        parameters_vec, self.network.parameters()
                    )
                    test_loss, test_acc = self.test(
                        test_loader, loss_criterion, test_size, self.hp.device
                    )
                    prune_losses.append(test_loss)
                    prune_accuracies.append(test_acc)
            return prune_losses, prune_accuracies
