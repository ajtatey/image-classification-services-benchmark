""" Renders main results plots. Assumes a csv datafile as input with columns:
[dataset, service, ablation, train_time, accuracy, latency]
"""

import os

import fire
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"

# These colors are picked from the previous benchmark post
# https://www.nyckel.com/blog/automl-benchmark-nyckel-google-huggingface/
nyckel_blue = "#4c72b0"
google_orange = "#dd8452"
huggingface_green = "#55a868"


def configure_matplotlib():
    plt.rcParams.update({"font.size": 14})

    sns.set(
        font="Franklin Gothic Book",
        rc={
            "axes.axisbelow": False,
            "axes.edgecolor": "lightgrey",
            "axes.facecolor": "None",
            "axes.grid": False,
            "axes.labelcolor": "dimgrey",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "white",
            "lines.solid_capstyle": "round",
            "patch.edgecolor": "w",
            "patch.force_edgecolor": True,
            "text.color": "dimgrey",
            "xtick.bottom": True,
            "xtick.color": "dimgrey",
            "xtick.direction": "out",
            "xtick.top": False,
            "ytick.color": "dimgrey",
            "ytick.direction": "out",
            "ytick.left": True,
            "ytick.right": False,
        },
    )


color_by_service = {
    "nyckel": nyckel_blue,
    "vertex": google_orange,
    "huggingface": huggingface_green,
    "aws": CB91_Pink,
}

services_in_order = ["nyckel", "vertex", "huggingface", "aws"]

symbols_by_dataset = {"cars": "P", "food": "X", "pets": "v", "intel": "o", "beans": "^", "clothing": "s", "xrays": "^"}

n_classes_by_dataset = {"cars": 196, "food": 101, "pets": 37, "intel": 6, "beans": 3, "clothing": 14, "xrays": 2}

ablations = [5, 20, 80, 320, 1280]

services = list(color_by_service.keys())

datasets = ["clothing", "food", "pets", "intel", "beans", "cars", "xrays"]

pretty_name_by_service = {"nyckel": "Nyckel", "vertex": "Google", "aws": "AWS", "huggingface": "Huggingface"}


def load_data(data_file_path: str):
    with open(data_file_path) as f:
        lines = f.readlines()

    data = {}
    for line in lines[1:]:
        dataset, service, ablation, train_time, accuracy, latency = line.split(",")
        if service not in data:
            data[service] = {}
        if dataset not in data[service]:
            data[service][dataset] = {"accuracy": [], "train_time": [], "ablation": [], "n_samples": []}
        data[service][dataset]["accuracy"].append(100 * float(accuracy))
        data[service][dataset]["train_time"].append(float(train_time))
        data[service][dataset]["ablation"].append(int(ablation))
        data[service][dataset]["n_samples"].append(int(ablation) * n_classes_by_dataset[dataset])
    return data


def render_accuracy_vs_traintime_as_lines(data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for service in services_in_order:
        for ditt, dataset in enumerate(data[service]):
            if ditt == 1:
                ax.plot(
                    data[service][dataset]["train_time"],
                    data[service][dataset]["accuracy"],
                    ".-",
                    color=color_by_service[service],
                    label=pretty_name_by_service[service],
                )
            else:
                ax.plot(
                    data[service][dataset]["train_time"],
                    data[service][dataset]["accuracy"],
                    ".-",
                    color=color_by_service[service],
                )

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Training time (s)")
    _ = ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Traintime", fontsize=18)

    fig.tight_layout()
    fig.savefig(f"result_plots/accuracy_vs_traintime_as_lines.png")
    fig.savefig(f"result_plots/accuracy_vs_traintime_as_lines.svg")


def render_accuracy_vs_traintime_as_symbols(data):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for service in services_in_order:
        for dataset in data[service]:
            label = f"{dataset} - {pretty_name_by_service[service]}"
            ax.plot(
                data[service][dataset]["train_time"][1],
                data[service][dataset]["accuracy"][1],
                symbols_by_dataset[dataset],
                markersize=8,
                color=color_by_service[service],
                label=label,
            )

    ax.set_xscale("log")
    ax.grid(which="minor", color="dimgrey", alpha=0.1)
    ax.grid(which="major", color="dimgrey", alpha=0.3)
    ax.legend(prop={"size": 8}, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.35))
    ax.set_xlabel("Training time (s)")
    _ = ax.set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig("result_plots/accuracy_vs_traintime_as_symbols.png")
    fig.savefig("result_plots/accuracy_vs_traintime_as_symbols.svg")


def render_runtime_vs_ablation_as_lines(data):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for service in services_in_order:
        for ditt, dataset in enumerate(data[service]):
            if ditt == 0:
                label = pretty_name_by_service[service]
            else:
                label = None
            ax.plot(
                data[service][dataset]["n_samples"],
                data[service][dataset]["train_time"],
                ".-",
                color=color_by_service[service],
                label=label,
            )

    ax.set_title("Traintime by Amount of Data", fontsize=18)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(which="minor", color="dimgrey", alpha=0.1)
    ax.grid(which="major", color="dimgrey", alpha=0.3)
    ax.set_xlabel("Training samples (count)")
    _ = ax.set_ylabel("Training time (s)")
    fig.tight_layout()
    fig.savefig("result_plots/runtime_vs_ablation_as_lines.png")
    fig.savefig("result_plots/runtime_vs_ablation_as_lines.svg")


def render_mean_accuracy_by_ablation_as_barplot(data):
    def mean_acc_by_service(service: str, ablation: int):
        accs = []
        for dataset in data[service]:
            for this_ablation, this_accuracy in zip(
                data[service][dataset]["ablation"], data[service][dataset]["accuracy"]
            ):
                if ablation == this_ablation:
                    accs.append(this_accuracy)
        return sum(accs) / len(accs)

    def std_acc_by_service(service: str, ablation: int):
        accs = []
        for dataset in data[service]:
            for this_ablation, this_accuracy in zip(
                data[service][dataset]["ablation"], data[service][dataset]["accuracy"]
            ):
                if ablation == this_ablation:
                    accs.append(this_accuracy)
        return np.std(accs) / np.sqrt(len(accs))

    ablations = [5, 20, 80, 320, 1280]
    barplot_data = {}
    for service in services:
        barplot_data[service] = {
            "mean": [mean_acc_by_service(service, abl) for abl in ablations],
            "std": [std_acc_by_service(service, abl) for abl in ablations],
        }

    N = len(ablations)

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # Width of a bar
    width = 0.2

    # Plotting
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for service, offset in zip(services, offsets):
        plt.bar(
            ind + offset,
            barplot_data[service]["mean"],
            width,
            label=pretty_name_by_service[service],
            yerr=barplot_data[service]["std"],
            color=color_by_service[service],
        )

    plt.title("Accuracy by ablation size", fontsize=18)
    plt.xlabel("Nbr. train sample per class")
    plt.ylabel("Accuracy (%)")
    plt.grid(which="minor", color="dimgrey", alpha=0.1)
    plt.grid(which="major", color="dimgrey", alpha=0.3)
    plt.ylim([50, 100])

    plt.xticks(ind, ablations)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("result_plots/mean_accuracy_by_ablation_as_barplot.png")
    plt.savefig("result_plots/mean_accuracy_by_ablation_as_barplot.svg")


def render_accuracy_by_dataset_as_barplot(data):
    barplot_data = {}
    for service in services:
        barplot_data[service] = [data[service][dataset]["accuracy"][-1] for dataset in datasets]

    N = len(datasets)

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # Width of a bar
    width = 0.2

    # Plotting
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for service, offset in zip(services, offsets):
        plt.bar(
            ind + offset,
            barplot_data[service],
            width,
            label=pretty_name_by_service[service],
            color=color_by_service[service],
        )

    plt.title("Accuracy by dataset & service", fontsize=18)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.grid(which="minor", color="dimgrey", alpha=0.1)
    plt.grid(which="major", color="dimgrey", alpha=0.3)
    plt.ylim([25, 100])

    plt.xticks(ind, datasets)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("result_plots/accuracy_by_dataset_as_barplot.png")
    plt.savefig("result_plots/accuracy_by_dataset_as_barplot.svg")


def render_traintime_by_dataset_as_barplot(data):
    barplot_data = {}
    for service in services:
        barplot_data[service] = [data[service][dataset]["train_time"][-1] for dataset in datasets]

    N = len(datasets)

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # Width of a bar
    width = 0.2

    # Plotting
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for service, offset in zip(services, offsets):
        plt.bar(
            ind + offset,
            barplot_data[service],
            width,
            label=pretty_name_by_service[service],
            color=color_by_service[service],
        )

    plt.title("Traintime by dataset & service", fontsize=18)
    plt.xlabel("Dataset")
    plt.ylabel("Traintime (s)")

    plt.yscale("log")

    plt.xticks(ind, datasets)
    plt.grid(which="minor", color="dimgrey", alpha=0.1)
    plt.grid(which="major", color="dimgrey", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("result_plots/traintime_by_dataset_as_barplot.png")
    plt.savefig("result_plots/traintime_by_dataset_as_barplot.svg")


def render_mean_traintime_vs_mean_accuracy(data):
    def mean_acc_by_service(service: str):
        accs = []
        for dataset in data[service]:
            accs.append(data[service][dataset]["accuracy"][-1])
        return sum(accs) / len(accs)

    def mean_traintime_by_service(service: str):
        traintimes = []
        for dataset in data[service]:
            traintimes.append(data[service][dataset]["train_time"][-1])
        return sum(traintimes) / len(traintimes)

    def std_acc_by_service(service: str):
        accs = []
        for dataset in data[service]:
            accs.append(data[service][dataset]["accuracy"][-1])
        return np.std(accs) / np.sqrt(len(accs))

    def std_traintime_by_service(service: str):
        traintimes = []
        for dataset in data[service]:
            traintimes.append(data[service][dataset]["train_time"][-1])
        return np.std(traintimes) / np.sqrt(len(traintimes))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for service in services:
        ax.errorbar(
            mean_traintime_by_service(service),
            mean_acc_by_service(service),
            yerr=std_acc_by_service(service),
            xerr=std_traintime_by_service(service),
            marker="o",
            label=pretty_name_by_service[service],
            markersize=12,
            color=color_by_service[service],
        )
        print(service, mean_traintime_by_service(service))
        print(service, mean_acc_by_service(service))

    ax.set_title("Accuracy vs. Traintime")
    ax.set_xscale("log")
    ax.legend(facecolor="white")
    ax.set_xlabel("Training time (s)")
    _ = ax.set_ylabel("Accuracy (%)")
    ax.grid(which="minor", color="dimgrey", alpha=0.1)
    ax.grid(which="major", color="dimgrey", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"result_plots/mean_traintime_vs_mean_accuracy.png")
    fig.savefig(f"result_plots/mean_traintime_vs_mean_accuracy.svg")


def render_selected(data):
    render_runtime_vs_ablation_as_lines(data)
    render_mean_accuracy_by_ablation_as_barplot(data)
    render_accuracy_by_dataset_as_barplot(data)
    render_traintime_by_dataset_as_barplot(data)
    render_mean_traintime_vs_mean_accuracy(data)


def main(data_file_path: str = "image-classification-benchmark-data.csv"):
    if not os.path.exists("result_plots"):
        os.makedirs("result_plots")
    configure_matplotlib()
    data = load_data(data_file_path)
    render_selected(data)


if __name__ == "__main__":
    fire.Fire(main)
