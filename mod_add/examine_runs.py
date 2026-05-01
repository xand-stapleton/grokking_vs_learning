# This script is meant to help me see what was in each of the runs


import inspect

from cluster_run_mod import ModularArithmeticDataset, TrainArgs
from prettytable import PrettyTable

from tools.data_objects import *


def files_summary(directory, variables):
    # Initialize a PrettyTable
    table = PrettyTable()
    table.field_names = ["Filename"] + variables

    # Loop through each file in the directory
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            print(filename)
            if (
                "grok_True" in dirpath or "grok_True" in filename
            ):  # or whatever file extension your files have
                file_path = os.path.join(dirpath, filename)
                # Load the object
                file_object = torch.load(file_path, map_location="cpu")
                # Access the attributes and add them to the table
                row = [filename]
                for var in variables:
                    try:
                        print("found attribute")
                        value = getattr(file_object.trainargs, var)
                        row.append(value)
                    except AttributeError:
                        print("no attribute")
                        row.append("N/A")  # If the attribute does not exist

                table.add_row(row)
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if (
                "grok_False" in dirpath or "grok_False" in filename
            ):  # or whatever file extension your files have
                file_path = os.path.join(dirpath, filename)
                # Load the object
                file_object = torch.load(file_path, map_location="cpu")
                # Access the attributes and add them to the table
                row = [filename]
                for var in variables:
                    try:
                        value = getattr(file_object.trainargs, var)
                        row.append(value)
                    except AttributeError:
                        row.append("N/A")  # If the attribute does not exist

                table.add_row(row)

    # Print the table
    print(table)


directory = "/Users/dmitrymanning-coe/Downloads/Cluster3"
current_dir = "/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/LossCurves/"
variables = [
    "lr",
    "hiddenlayers",
    "weight_decay",
    "weight_multiplier",
    "grok",
    "epochs",
]
files_summary(directory=directory, variables=variables)
files_summary(directory=current_dir, variables=variables)
