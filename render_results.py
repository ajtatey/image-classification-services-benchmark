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

# These two colors are picked from the previous benchmark post
# https://www.nyckel.com/blog/automl-benchmark-nyckel-google-huggingface/
google_orange = "#dd8452" 
huggingface_green = "#55a868" 

nyckel_blue = "#1F73C3" # From the nyckel favicon
huggingface_yellow = "#FECA19" # Yellow from inside of huggingface logo
amazon_orange = "#FD8608" # Orange from arrow in amazon logo
google_green = "#2DA541" # Green from the google logo
face_color = "#B0EEFF", #from clouds in bounding box box post hero
#face_color = "#AADEFC", #from bounding box post graph
#face_color = "#BDD4FE", #from semantic search post (light part of lens)


def configure_matplotlib():
    plt.rcParams.update({"font.size": 14})

    sns.set(
        font="Fuzzy Bubbles",
        font_scale=1,
        rc={
            "axes.axisbelow": True,
            "axes.edgecolor": "k",
            "axes.facecolor": "None",
            "axes.grid": False,
            "axes.labelcolor": "k",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.facecolor": "#B0EEFF", #Using face_color here doesn't work for some reason
            "lines.solid_capstyle": "round",
            "patch.edgecolor": "w",
            "patch.force_edgecolor": True,
            "text.color": "k",
            "xtick.bottom": True,
            "xtick.color": "k",
            "xtick.direction": "out",
            "xtick.top": False,
            "ytick.color": "k",
            "ytick.direction": "out",
            "ytick.left": True,
            "ytick.right": False,
        },
    )


color_by_service = {
    "nyckel": nyckel_blue,
    "vertex": google_green,
    "huggingface": huggingface_yellow,
    "aws": amazon_orange,
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

def render_accuracy_by_ablation_and_dataset_as_barplot(data):
    # For each dataset, plot accuracy by ablation size and service on a barplot
    # Tile all these barplots in two columns in a figure
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    fig.suptitle("Accuracy by ablation size for each dataset", fontsize=18)
    subplots = []
    subplots.append(plt.subplot2grid((4, 4), (0, 0), colspan=2))
    subplots.append(plt.subplot2grid((4, 4), (0, 2), colspan=2))
    subplots.append(plt.subplot2grid((4, 4), (1, 0), colspan=2))
    subplots.append(plt.subplot2grid((4, 4), (1, 2), colspan=2))
    subplots.append(plt.subplot2grid((4, 4), (2, 0), colspan=2))
    subplots.append(plt.subplot2grid((4, 4), (2, 2), colspan=2))
    subplots.append(plt.subplot2grid((4, 4), (3, 1), colspan=2))

    for idx, dataset in enumerate(datasets):
        bars = render_accuracy_by_ablation_for_dataset_as_barplot(data, dataset, subplots[idx])

    legendAx = plt.subplot2grid((4,4), (3,3), colspan=1)
    legendAx.axis("off")
    service_names = []
    for service in services:
        service_names.append(pretty_name_by_service[service])
    legendAx.legend(bars, service_names, loc="center")

    plt.tight_layout()
    plt.savefig(f"result_plots/accuracy_by_ablation_all_as_barplot.png")
    plt.savefig(f"result_plots/accuracy_by_ablation_all_as_barplot.svg")
 

def render_accuracy_by_ablation_for_dataset_as_barplot(data, dataset, plot):
    def acc_by_dataset_and_service(service: str, dataset: str, ablation: int):
        for this_ablation, this_accuracy in zip(data[service][dataset]["ablation"], data[service][dataset]["accuracy"]):
            if ablation == this_ablation:
                return this_accuracy
        return 0

    ablations = [5, 20, 80, 320, 1280]
    barplot_data = {}
    for service in services:
        barplot_data[service] = {
            "acc": [acc_by_dataset_and_service(service, dataset, abl) for abl in ablations],
        }

    N = len(ablations)

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    #plt.figure(figsize=(10, 5))

    # Width of a bar
    width = 0.2

    # Plotting
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    bars = []
    for service, offset in zip(services, offsets):
        bars.append(plot.bar(
            ind + offset,
            barplot_data[service]["acc"],
            width,
            label=pretty_name_by_service[service],
            color=color_by_service[service],
            alpha=1,
        ))

    plot.set_title(dataset, fontsize=12)
    plot.set_ylim([0, 100])
    plot.set_xticks(ind, ablations)
    return bars


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
            alpha=1,
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
            alpha=1,
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
            alpha=1,
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


def render_mean_traintime_vs_mean_accuracy(data, include_legend=True, output_file_name="mean_traintime_vs_mean_accuracy"):
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
            elinewidth=2.5,
        )
        print(service, mean_traintime_by_service(service))
        print(service, mean_acc_by_service(service))

    ax.set_title("Accuracy vs. Traintime", fontsize=18, fontweight="bold")
    ax.set_xscale("log")
    if(include_legend):
        ax.legend(facecolor="#B0EEFF")
    ax.set_xlabel("Training time (s)")
    _ = ax.set_ylabel("Accuracy (%)")
    ax.grid(which="minor", color="dimgrey", alpha=0.1)
    ax.grid(which="major", color="dimgrey", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"result_plots/{output_file_name}.png")
    fig.savefig(f"result_plots/{output_file_name}.svg")

def render_latency_and_throughput():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    render_throughput(ax[1])
    render_latency(ax[0])

    fig.tight_layout()
    fig.savefig("result_plots/throughput_latency.png")
    fig.savefig("result_plots/throughput_latency.svg")

def render_latency(plot):
    with open ("latency.csv") as f:
        lines = f.readlines()

    service_colors=[]
    boxes=[]
    for line in lines[1:]:
        service, min, p25, p50, p75, max = line.split(",")
        boxes.append({
            'label' : pretty_name_by_service[service],
            'whislo': int(min),    # Bottom whisker position
            'q1'    : int(p25),    # First quartile (25th percentile)
            'med'   : int(p50),    # Median         (50th percentile)
            'q3'    : int(p75),    # Third quartile (75th percentile)
            'whishi': int(max),    # Top whisker position
            'fliers': []        # Outliers
        })
        service_colors.append(color_by_service[service])

    bplot = plot.bxp(boxes, showfliers=False, patch_artist=True)
    for patch, color in zip(bplot['boxes'], service_colors):
        patch.set_facecolor(color)
    plot.set_ylabel("Latency (ms)")
    plot.set_title("Inference latency", fontsize=12)
    plot.grid(which="major", color="dimgrey", alpha=0.3)

def render_throughput(plot):
    # Throughput values (in rps) are in the data file throughput.csv
    # The header row describes the columns
    # The columns are service,througput(rps)
    # Read in the file and plot the values as a bar graph
    with open("image-classification-throughputs.csv") as f:
        lines = f.readlines()

    N = len(lines) - 1

    # Position of bars on x-axis
    ind = np.arange(N)

    # Width of a bar
    width = 0.2

    service_names=[]
    for line in lines[1:]:
        service, time_for_1000_requests = line.split(",")
        throughput = 1000 / float(time_for_1000_requests)
        service_names.append(pretty_name_by_service[service])
        plot.bar(
            service,
            float(throughput),
            width,
            label=pretty_name_by_service[service],
            color=color_by_service[service],
            alpha=1,
        )

    plot.set_title("Inference throughput at 10 concurrent requests", fontsize=12)
    plot.set_ylabel("Requests per second")
    plot.grid(which="major", color="dimgrey", alpha=0.3)
    plot.set_xticks(ind, service_names)

def render_devex():
    with open("image-classification-usability.csv") as f:
        lines = f.readlines()

    N = len(lines) - 1

    # Position of bars on x-axis
    ind = np.arange(N)

    # Width of a bar
    width = 0.2

    plt.figure(figsize=(10, 5))

    service_names=[]
    for line in lines[1:]:
        service, data, expertise, training, invoke, docs = line.split(",")
        total = int(data)+int(expertise)+int(training)+int(invoke)+int(docs)
        service_names.append(pretty_name_by_service[service])
        plt.bar(
            service,
            total,
            width,
            label=pretty_name_by_service[service],
            color=color_by_service[service],
            alpha=1,
        )

    plt.title("Developer experience scores by service (higher is better)", fontsize=18)
    plt.ylabel("Total score")
    plt.grid(which="major", color="dimgrey", alpha=0.3)
    plt.xticks(ind, service_names)
    plt.savefig("result_plots/devex.png")
    plt.savefig("result_plots/devex.svg")

def render_selected(data):
    render_runtime_vs_ablation_as_lines(data)
    render_mean_accuracy_by_ablation_as_barplot(data)
    render_accuracy_by_ablation_and_dataset_as_barplot(data)
    render_accuracy_by_dataset_as_barplot(data)
    render_traintime_by_dataset_as_barplot(data)
    render_mean_traintime_vs_mean_accuracy(data, True)
    render_mean_traintime_vs_mean_accuracy(data, False, "mean_traintime_vs_mean_accuracy_no_legend")
    render_latency_and_throughput()
    render_devex()


def main(data_file_path: str = "image-classification-benchmark-data.csv"):
    if not os.path.exists("result_plots"):
        os.makedirs("result_plots")
    configure_matplotlib()
    data = load_data(data_file_path)
    render_selected(data)


if __name__ == "__main__":
    fire.Fire(main)
