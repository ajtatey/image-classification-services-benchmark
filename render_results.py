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

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]


def configure_matplotlib():
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=color_list)
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


color_by_service = {"nyckel": CB91_Blue, "vertex": CB91_Pink, "aws": CB91_Green, "huggingface": CB91_Amber}

symbols_by_dataset = {"cars": "P", "food": "X", "pets": "v", "intel": "o", "beans": "^", "clothing": "s", "xrays": "^"}

n_classes_by_dataset = {"cars": 196, "food": 101, "pets": 37, "intel": 6, "beans": 3, "clothing": 14, "xrays": 2}

ablations = [5, 20, 80, 320, 1280]

services = list(color_by_service.keys())

datasets = ["clothing", "food", "pets", "intel", "beans", "cars", "xrays"]

services_legend = {"nyckel": "Nyckel", "vertex": "Google", "aws": "AWS", "huggingface": "Huggingface"}


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
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xscale("log")
    for service in data:
        for ditt, dataset in enumerate(data[service]):
            if ditt == 1:
                ax.plot(
                    data[service][dataset]["train_time"],
                    data[service][dataset]["accuracy"],
                    ".-",
                    color=color_by_service[service],
                    label=service,
                )
            else:
                ax.plot(
                    data[service][dataset]["train_time"],
                    data[service][dataset]["accuracy"],
                    ".-",
                    color=color_by_service[service],
                )

    ax.legend()
    ax.set_xlabel("Training time (s)")
    _ = ax.set_ylabel("Accuracy (%)")

    fig.tight_layout()
    fig.savefig("result_plots/accuracy_vs_traintime_as_lines.png")


def render_accuracy_vs_traintime_as_symbols(data):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xscale("log")
    for service in data:
        for dataset in data[service]:
            label = f"{dataset} - {service}"
            ax.plot(
                data[service][dataset]["train_time"][1],
                data[service][dataset]["accuracy"][1],
                symbols_by_dataset[dataset],
                markersize=8,
                color=color_by_service[service],
                label=label,
            )

    ax.grid(which="minor", color="dimgrey", alpha=0.1)
    ax.grid(which="major", color="dimgrey", alpha=0.3)
    ax.legend(prop={"size": 8}, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.35))
    ax.set_xlabel("Training time (s)")
    _ = ax.set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig("result_plots/accuracy_vs_traintime_as_symbols.png")


def render_runtime_vs_ablation_as_lines(data):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_yscale("log")
    for service in data:
        for ditt, dataset in enumerate(data[service]):
            if ditt == 0:
                label = service
            else:
                label = None
            ax.plot(
                data[service][dataset]["n_samples"],
                data[service][dataset]["train_time"],
                ".-",
                color=color_by_service[service],
                label=label,
            )

    ax.legend()
    ax.set_xlabel("Training samples (count)")
    _ = ax.set_ylabel("Training time (s)")
    fig.tight_layout()
    fig.savefig("result_plots/runtime_vs_ablation_as_lines.png")


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
            label=service,
            yerr=barplot_data[service]["std"],
            color=color_by_service[service],
        )

    plt.xlabel("Train sample per class")
    plt.ylabel("Accuracy (%)")
    plt.title("Mean accurcy by ablation size")

    plt.ylim([50, 100])

    plt.xticks(ind, ablations)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("result_plots/render_mean_accuracy_by_ablation_as_barplot.png")


def render_accuracy_by_dataset_as_barplot(data, ablation_index):
    barplot_data = {}
    for service in services:
        barplot_data[service] = [data[service][dataset]["accuracy"][ablation_index] for dataset in datasets]

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
            label=service,
            color=color_by_service[service],
        )

    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy by dataset & service")

    plt.ylim([25, 100])

    plt.xticks(ind, datasets)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"result_plots/accuracy_by_ablation_as_barplot_{ablation_index=}.png")


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
            label=service,
            color=color_by_service[service],
        )

    plt.xlabel("Dataset")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy by dataset & service")

    plt.ylim([25, 100])

    plt.xticks(ind, datasets)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("result_plots/accuracy_by_dataset_as_barplot.png")


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
            label=service,
            color=color_by_service[service],
        )

    plt.xlabel("Dataset")
    plt.ylabel("Traintime (s)")
    plt.title("Traintime by dataset & service")

    plt.yscale("log")

    plt.xticks(ind, datasets)

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("result_plots/traintime_by_dataset_as_barplot.png")


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
    ax.set_xscale("log")
    for service in services:
        ax.errorbar(
            mean_traintime_by_service(service),
            mean_acc_by_service(service),
            yerr=std_acc_by_service(service),
            xerr=std_traintime_by_service(service),
            marker="o",
            label=services_legend[service],
            markersize=12,
            color=color_by_service[service],
        )
        print(service, mean_traintime_by_service(service))
        print(service, mean_acc_by_service(service))

    ax.legend(facecolor="white")
    ax.set_xlabel("Training time (s)")
    _ = ax.set_ylabel("Accuracy (%)")
    ax.grid(which="minor", color="dimgrey", alpha=0.1)
    ax.grid(which="major", color="dimgrey", alpha=0.3)
    fig.tight_layout()
    fig.savefig("result_plots/mean_traintime_vs_mean_accuracy.png")


def main(data_file_path: str = "image-classification-benchmark-data.csv"):
    if not os.path.exists("result_plots"):
        os.makedirs("result_plots")
    configure_matplotlib()
    data = load_data(data_file_path)
    render_accuracy_vs_traintime_as_lines(data)
    render_accuracy_vs_traintime_as_symbols(data)
    render_runtime_vs_ablation_as_lines(data)
    render_mean_accuracy_by_ablation_as_barplot(data)
    render_accuracy_by_dataset_as_barplot(data)
    render_traintime_by_dataset_as_barplot(data)
    render_mean_traintime_vs_mean_accuracy(data)


if __name__ == "__main__":
    fire.Fire(main)
