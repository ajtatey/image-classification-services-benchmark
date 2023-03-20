import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem

DATASETS = ["beans", "cars", "food", "intel", "pets", "clothing", "xrays"]
SERVICES = ["nyckel", "huggingface", "aws", "vertex"]
ABLATIONS = [5, 20, 80, 320, 1280]

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
colors = list(color_by_service.values())
services = list(color_by_service.keys())


def get_accuracies():
    accuracies = []
    for dataset in DATASETS:
        if os.path.exists(f"data/{dataset}/results"):
            for file in os.listdir(f"data/{dataset}/results"):
                if file != ".DS_Store":
                    with open(f"data/{dataset}/results/{file}") as csvfile:
                        accuracy = 0
                        total = 0
                        reader = csv.reader(csvfile)
                        for row in reader:
                            if str(row[1]) == str(row[2]):
                                accuracy += 1
                            total += 1
                        accuracies.append(accuracy / total)
                        print(f"Accuracy for {file}: {accuracy/total}")


def get_combined_accuracies():
    combined_accuracies = np.zeros((len(SERVICES), len(ABLATIONS)))
    combined_errors = np.zeros((len(SERVICES), len(ABLATIONS)))

    for service_count, service in enumerate(SERVICES):
        for ablation_count, ablation in enumerate(ABLATIONS):
            ablation_means = np.zeros(len(DATASETS))
            for dataset_count, dataset in enumerate(DATASETS):
                if os.path.exists(f"data/{dataset}/results"):
                    for file in os.listdir(f"data/{dataset}/results"):
                        if file != ".DS_Store":
                            if service in file and str(ablation) in file:
                                with open(f"data/{dataset}/results/{file}") as csvfile:
                                    accuracy = 0
                                    total = 0
                                    reader = csv.reader(csvfile)
                                    for row in reader:
                                        if str(row[1]) == str(row[2]):
                                            accuracy += 1
                                        total += 1
                                if total != 0:
                                    print(f"Accuracy for {file}: {accuracy/total}")
                                    ablation_means[dataset_count] = accuracy / total
            # print(ablation_means)
            ablation_means[ablation_means == 0] = np.nan
            # print(ablation_means)
            # print(np.nanmean(ablation_means))
            combined_accuracies[service_count, ablation_count] = np.nanmean(ablation_means)
            combined_errors[service_count, ablation_count] = sem(ablation_means, nan_policy="omit")

    _, ax = plt.subplots()
    ax.set_xscale("log")
    ax.errorbar(
        ABLATIONS,
        combined_accuracies[0],
        yerr=combined_errors[0],
        fmt="o",
        label="Vertex",
        linestyle="dotted",
        capsize=6,
    )
    ax.errorbar(
        ABLATIONS, combined_accuracies[1], yerr=combined_errors[1], fmt="o", label="AWS", linestyle="dotted", capsize=6
    )
    ax.errorbar(
        ABLATIONS,
        combined_accuracies[2],
        yerr=combined_errors[2],
        fmt="o",
        label="Huggingface",
        linestyle="dotted",
        capsize=6,
    )
    ax.errorbar(
        ABLATIONS,
        combined_accuracies[3],
        yerr=combined_errors[3],
        fmt="o",
        label="Nyckel",
        linestyle="dotted",
        capsize=6,
    )

    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()


def get_latencies():
    for latency_count, service in enumerate(SERVICES):
        latencies = []
        for dataset in DATASETS:
            if os.path.exists(f"data/{dataset}/results"):
                for file in os.listdir(f"data/{dataset}/results"):
                    if service == "huggingface":
                        service = "hg"
                    if service in file:
                        with open(f"data/{dataset}/results/{file}") as csvfile:
                            reader = csv.reader(csvfile)
                            latencies = []

                            for row in reader:
                                if service in ["aws", "vertex"]:
                                    latencies.append(float(row[4]))
                                else:
                                    latencies.append(float(row[4][-8:-1]))
                        print(f"Latency for {file}: {sum(latencies)/len(latencies)}")


def get_combined_latencies():
    all_latencies = [[], [], [], []]
    for latency_count, service in enumerate(SERVICES):
        latencies = []
        for dataset in DATASETS:
            if os.path.exists(f"data/{dataset}/results"):
                for file in os.listdir(f"data/{dataset}/results"):
                    if service == "huggingface":
                        service = "hg"
                    if service in file:
                        with open(f"data/{dataset}/results/{file}") as csvfile:
                            reader = csv.reader(csvfile)
                            for row in reader:
                                if service in ["aws", "vertex"]:
                                    latencies.append(float(row[4]))
                                else:
                                    latencies.append(float(row[4][-8:-1]))
        print(f"Latency for {service}: {sum(latencies)/len(latencies)}")
        all_latencies[latency_count].extend(latencies)
    # create boxplots for each without outliers
    _, (ax1, ax2) = plt.subplots(1, 2)
    bplot = ax1.boxplot(all_latencies, showfliers=False, patch_artist=True)
    ax1.set_xticklabels(SERVICES)
    ax1.set_title("Latency per invoke")
    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("Dataset")
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
    for median in bplot["medians"]:
        median.set_color("black")

    services = []
    throughputs = []
    with open("image-classification-throughputs.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "service":
                continue
            services.append(row[0])
            print(row[1])
            throughputs.append(float(row[1]))

    ax2.bar(services, throughputs, color=colors)
    ax2.set_title("Throughput - 1k concurrent invokes")
    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Dataset")
    plt.show()


def usability_metrics():
    services = []
    usability_scores = []
    with open("image-classification-usability.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "service":
                continue
            services.append(row[0])
            usability_scores.append(int(row[1]))

    _, ax = plt.subplots()
    ax.bar(services, usability_scores, color=colors)
    ax.set_title("Usability Score")
    ax.set_ylabel("Score (out of 25)")
    ax.set_xlabel("Dataset")
    plt.show()


if __name__ == "__main__":
    get_combined_accuracies()
    get_accuracies()
    get_latencies()
    configure_matplotlib()
    get_combined_latencies()
    usability_metrics()
