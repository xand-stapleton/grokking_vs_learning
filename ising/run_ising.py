# Need to add path to run in cluster also
import datetime
import inspect
import os
import sys
from os import getpid
from pathlib import Path

# This always does the imports relative to the modadd folder.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import inspect
import itertools
import time

import torch
from tqdm import tqdm
import yaml

from data_objects import seed_average_onerun
from datasets.ising_dataset import create_ising_dataset
from models.architechtures import CNN_Rect
from models.constructor import Model
from tools.config_method_parser import (
    str_to_activation,
    str_to_loss,
    str_to_optimiser,
)
from tools.train_args import TrainArgs


def main():
    # Ha! MPS Is slower than cpu! Maybe just for modadd
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    else:
        device = torch.device("cpu")

    # This allows you to modify any of the yaml configs from the command line.
    parser = argparse.ArgumentParser(
        description="Update YAML configuration values."
    )

    # Grok argument
    parser.add_argument(
        "--grok",
        dest="grok",
        action="store_true",
        help="Enable grok mode.",
    )
    parser.add_argument(
        "--no-grok",
        dest="grok",
        action="store_false",
        help="Disable grok mode.",
    )
    parser.set_defaults(grok=True)  # Default to --grok

    known_args, unknown_args = parser.parse_known_args()

    # Determine which YAML file to load
    config_file = (
        "configs/ising_config_grok.yaml"
        if known_args.grok
        else "configs/ising_config_nogrok.yaml"
    )

    # Load YAML file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Update config with command-line arguments if provided
    for arg in unknown_args:
        if "=" in arg:
            # Remove optional -- at beginning of arg
            arg = arg.lstrip("-")
            key, value = arg.split("=", 1)
            print(f"Overriding default {key} with {value}")
            config[key] = value  # Update config with command-line arguments
        else:
            raise ValueError(
                f"Invalid argument format: '{arg}'. Use key=value."
            )

    evaluate_dic = {
        "euclidcosine": False,
        "iprs": False,
        "norms": False,
        "linear_decomposition": False,
        "decile_steps": False,
        "fims": False,
    }

    activation = str_to_activation(config["activation"])
    all_seeds_same = bool(config["all_seeds_same"])
    bs = int(config["bs"])
    cluster_arg = bool(config["cluster_arg"])
    conv_channels = tuple(config["conv_channels"])
    data_seed_end = int(config["data_seed_end"])
    data_seed_start = int(config["data_seed_start"])
    desc = str(config["desc"])
    dropout_prob = float(config["dropout_prob"])
    epochs = int(config["epochs"])
    fisher_bs = int(config["fisher_bs"])
    fisher_seed_end = int(config["fisher_seed_end"])
    fisher_seed_start = int(config["fisher_seed_start"])
    init_seed_end = int(config["init_seed_end"])
    init_seed_start = int(config["init_seed_start"])
    learning_rate = float(config["learning_rate"])
    loss_criterion = str(config["loss_criterion"])
    loss_criterion = str_to_loss(loss_criterion)
    optimiser_name = str(config["optimiser_name"])
    optimiser = str_to_optimiser(optimiser_name)
    save_interval = int(config["save_interval"])
    sgd_seed_end = int(config["sgd_seed_end"])
    sgd_seed_start = int(config["sgd_seed_start"])
    test_size = int(config["test_size"])
    train_fraction = float(config["train_fraction"])
    train_size = int(config["train_size"])
    weight_decay = float(config["weight_decay"])
    weight_multiplier = float(config["weight_multiplier"])

    if type(config["hiddenlayers_input"]) == list:
        hiddenlayers = list(
            config["hiddenlayers_input"]
        )  # Note list([1,2,3]) is the same list
    elif type(config["hiddenlayers_input"]) == str:
        hiddenlayers = [int(i) for i in config["hiddenlayers_input"].split(",")]

    else:
        raise RuntimeError("Invalid hidden layers input.")

    data_seeds = [i for i in range(data_seed_start, data_seed_end)]
    sgd_seeds = [i for i in range(sgd_seed_start, sgd_seed_end)]
    init_seeds = [i for i in range(init_seed_start, init_seed_end)]
    fisher_seeds = [i for i in range(fisher_seed_start, fisher_seed_end)]

    date = datetime.datetime.now().replace(microsecond=0).isoformat()
    run_id = f"{date}_pid-{getpid()}"

    if cluster_arg:
        root_dir = str(config["save_loc_cluster_d"])
    else:
        root_dir = str(config["save_loc_local_d"])

    root_dir = (
        root_dir
        + "/"
        + f"hiddenlayer_{hiddenlayers}_desc_{desc}_wm_{weight_multiplier}_{run_id}"
    )

    print(
        f" seeds: {data_seeds}, sgd_seeds: {sgd_seeds}, init_seeds {init_seeds}"
    )
    print(f"device: {device}")
    print(f" wd: {weight_decay}")

    os.makedirs(root_dir, exist_ok=True)
    print("Save location: ", str(root_dir))

    if all_seeds_same:
        seeds = [(i, i, i, i) for i in data_seeds]
    else:
        seeds = list(
            itertools.product(data_seeds, sgd_seeds, init_seeds, fisher_seeds)
        )

    for seed_triple in (pbar := tqdm(seeds, desc="Seeds")):
        pbar.set_description(f"Seed: {seed_triple}")

        data_seed, sgd_seed, init_seed, fisher_seed = seed_triple

        # Define a data_run_object. Save this instead of the dictionary.
        print('Using weight multiplier: ', weight_multiplier)
        args = TrainArgs(
            epochs=epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
            weight_multiplier=weight_multiplier,
            dropout_prob=dropout_prob,
            data_seed=data_seed,
            sgd_seed=sgd_seed,
            init_seed=init_seed,
            fisher_seed=fisher_seed,
            fisher_bs=fisher_bs,
            device=device,
            hiddenlayers=hiddenlayers,
            conv_channels=conv_channels,
            test_size=test_size,
            train_size=train_size,
            train_fraction=train_fraction,
            P=None,
            batch_size=bs,
            loss_criterion=loss_criterion,
            root_dir=root_dir,
        )

        params_dic = {
            "weight_decay": weight_decay,
            "weight_multiplier": weight_multiplier,
            "learning_rate": learning_rate,
            "hidden_layers": hiddenlayers,
            "conv_channels": conv_channels,
            "train_size": train_size,
            "test_size": test_size,
            "dropout_p": dropout_prob,
        }
        save_object = seed_average_onerun(
            data_seed=args.data_seed,
            sgd_seed=args.sgd_seed,
            init_seed=args.init_seed,
            params_dic=params_dic,
        )
        save_object.trainargs = args

        save_object.start_time = int(time.time())

        train_loader, test_loader = create_ising_dataset(
            data_seed=data_seed,
            train_size=train_size,
            test_size=test_size,
            batch_size=train_size,
            cluster=cluster_arg,
        )

        _, fisher_loader = create_ising_dataset(
            data_seed=fisher_seed,
            train_size=train_size,
            test_size=test_size,
            batch_size=fisher_bs,
            cluster=cluster_arg,
            test_shuffle=True,
        )

        save_object.train_loader = train_loader
        save_object.test_loader = test_loader
        save_object.fisher_loader = fisher_loader

        input_dims = next(enumerate(train_loader))[1][0].shape[
            -2:
        ]  # 28 x 28 array of pixel values in MNIST (16x16 in Ising data)
        output_dim = 2  # (2 diff phases for Ising)
        bs = 200

        network = CNN_Rect(
            input_dims=input_dims,
            output_size=output_dim,
            input_channels=1,
            conv_channels=conv_channels,
            hidden_widths=hiddenlayers,
            activation=activation,
        )

        save_name = f"grok_dataseed_{data_seed}_sgdseed_{sgd_seed}_initseed_{init_seed}_wd_{weight_decay}_wm_{weight_multiplier}_time_{int(time.time())}"
        save_path = f"{root_dir}/{save_name}"

        # Make the parent directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        model = Model(
            args,
            optimiser,
            network,
            train_loader,
            test_loader,
            fisher_loader,
            save_path,
        )

        # Check if the instance has the _init_args attribute
        if hasattr(model.network, "_init_args"):
            # had to call 'args' 'margs' because of the other args!
            # Extract the stored initialization arguments
            margs, kwargs = model.network._init_args

            # Reconstruct the configuration dictionary
            config_dict = {
                **dict(
                    zip(
                        inspect.signature(
                            model.network.__class__.__init__
                        ).parameters,
                        margs,
                    )
                ),
                **kwargs,
            }

            if "self" in config_dict:
                config_dict.pop("self")
        else:
            raise RuntimeError(
                "The instance was created before the decorator was added and does not have initialization arguments stored."
            )

        save_object.modelinstance = copy.deepcopy(model)
        save_object.modelclass = model.__class__
        save_object.modelconfig = config_dict
        print(config_dict)

        model.train(
            epochs=args.epochs,
            save_interval=save_interval,
            train_loader=train_loader,
            test_loader=test_loader,
            one_run_object=save_object,
            loss_criterion=args.loss_criterion,
            device=device,
            evaluate_dic=evaluate_dic,
            fisher_loader=fisher_loader,
        )

        with open(f"{save_path}/save_blob.pt", "wb") as dill_file:
            torch.save(save_object, dill_file)

        layer_indices = list(
            range(sum(1 for _ in model.network.named_parameters()))
        )
        layer_indices = [None] + layer_indices

        for layer in layer_indices:
            prune_losses, prune_accuracies = model.fisher_prune(
                config["cutoff_beginning"],
                config["cutoff_end"],
                config["cutoff_num"],
                fisher_loader,
                test_loader,
                loss_criterion,
                logspace=True,
                prune_by_cutoff=False,
                prune_layer=layer,
            )

            with open(
                f"{save_path}/fim_prune_loss_acc_{layer}-pruned.pt", "wb"
            ) as dill_file:
                torch.save((prune_losses, prune_accuracies), dill_file)


if __name__ == "__main__":
    main()
