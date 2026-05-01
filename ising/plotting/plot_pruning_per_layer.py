import re
from pathlib import Path

import numpy as np
import torch
from simple_plot import create_pruning_plot, value_to_rgb, pick_colours
import argparse
from pathlib import Path
from natsort import natsorted
import plotly.io as pio   
pio.kaleido.scope.mathjax = None


def main(args, layer):
    parent_directory = args.file_path

    loss_plot_data = {}
    acc_plot_data = {}
    weight_multipliers_and_paths = get_weight_multiplier_paths(parent_directory)
    for wm_path_pair in weight_multipliers_and_paths:
        wm, wm_path = wm_path_pair
        wm_seed_avg_loss, wm_seed_avg_acc = aggregate_seed_metrics_from_files(
            wm_path,
            layer,
        )

        loss_plot_data = loss_plot_data | {
            f"{wm}": {
                "x": 100
                * np.arange(len(wm_seed_avg_acc) + 1)
                / len(wm_seed_avg_loss),
                "y": wm_seed_avg_loss,
                "color": pick_colours(float(wm), 0.1, 10),
            }
        }
        acc_plot_data = acc_plot_data | {
            f"{wm}": {
                "x": 100
                * np.arange(len(wm_seed_avg_acc) + 1)
                / len(wm_seed_avg_acc),
                "y": wm_seed_avg_acc,
                "color": pick_colours(float(wm), 0.1, 10),
            }
        }

    # Generate the plot and save it as a PDF
    fig = create_pruning_plot(
        loss_plot_data,
        title="",
        axis_labels=("% parameters pruned", "Loss"),
        text_sizes={"title": 16, "axis_labels": 20, "legend": 20},
        save_path=f"layer_{layer}_pruned_pruning_loss_curves.pdf",
    )
    fig.show()

    # Generate the plot and save it as a PDF
    fig = create_pruning_plot(
        acc_plot_data,
        title="",
        axis_labels=("% parameters pruned", "Accuracy"),
        text_sizes={"title": 16, "axis_labels": 20, "legend": 20},
        save_path=f"layer_{layer}_pruned_global_pruning_acc_curves.pdf",
    )
    fig.show()


def get_weight_multiplier_paths(parent_dir):
    """
    Extracts weight_multiplier and directory paths from a given parent directory.

    Under each directory all the seed directories.

    Args:
        parent_dir (str or Path): Path to the parent directory containing subdirectories.

    Returns:
        list: A list of tuples in the form (weight_multiplier, path).
    """
    parent_path = Path(parent_dir)
    result = []

    # Define the regex pattern to extract the weight_multiplier
    pattern = re.compile(r"hiddenlayer_\[\d+\]_desc_ising_wm_(\d+\.\d+)_.*")

    for dir_path in parent_path.iterdir():
        if dir_path.is_dir():  # Check if it's a directory
            match = pattern.search(dir_path.name)
            if match:
                weight_multiplier = match.group(1)
                result.append((weight_multiplier, str(dir_path)))

    return natsorted(result, key=lambda x: x[0])


def aggregate_seed_metrics_from_files(base_dir, layer):
    """
    Aggregates losses and accuracies across seeds from specific metric files.

    This function searches recursively within a specified directory for files named
    "fim_prune_loss_acc.pt". Each file is expected to store a tuple containing
    loss and accuracy values. The function collects these values across all seeds,
    computes their averages, and returns them.

    Args:
        base_dir (str or Path): The base directory to search for metric files.

    Returns:
        tuple: A tuple containing:
            - seed_average_loss (numpy.ndarray): The average loss across seeds.
            - seed_average_accuracy (numpy.ndarray): The average accuracy across seeds.

    Assumes:
        - Each file named "fim_prune_loss_acc.pt" stores a tuple of (loss, accuracy).
        - Loss and accuracy can be averaged directly using numpy arrays.
    """
    # Initialize lists to store all losses and accuracies
    all_seed_losses = []
    all_seed_accuracies = []

    # Search for all fim_prune_loss_acc.pt files in subdirectories (one for each seed)
    for file_path in Path(base_dir).rglob(
        f"fim_prune_loss_acc_{layer}-pruned.pt"
    ):
        loss, acc = torch.load(
            file_path
        )  # Assuming the file stores a tuple (loss, acc)
        all_seed_losses.append(loss)
        all_seed_accuracies.append(acc)

    # Convert the lists to arrays to facilitate averaging
    seed_average_loss = np.array(all_seed_losses).mean(axis=0)
    seed_average_accuracy = np.array(all_seed_accuracies).mean(axis=0)

    return seed_average_loss, seed_average_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file path.")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=Path.cwd(),
        help="Path to the file (default: current working directory)",
    )
    args = parser.parse_args()
    for layer in list(range(8)) + [None]:
        main(args, layer)
