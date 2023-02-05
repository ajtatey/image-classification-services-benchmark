import csv
import os
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["beans", "cars", "food", "intel", "pets", "clothing", "xrays"]
SERVICES = ["vertex", "aws", "hg", "nyckel"]
ABLATIONS = [5, 20, 80, 320, 1280]


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

    for service_count, service in enumerate(SERVICES):
        for ablation_count, ablation in enumerate(ABLATIONS):
            accuracy = 0
            total = 0
            for dataset in DATASETS:
                if os.path.exists(f"data/{dataset}/results"):
                    for file in os.listdir(f"data/{dataset}/results"):
                        if service in file and str(ablation) in file:
                            with open(f"data/{dataset}/results/{file}") as csvfile:
                                reader = csv.reader(csvfile)
                                for row in reader:
                                    if str(row[1]) == str(row[2]):
                                        accuracy += 1
                                    total += 1
            if total != 0:
                combined_accuracies[service_count, ablation_count] = accuracy / total
    print(combined_accuracies)

    """_, ax = plt.subplots()
    ax.errorbar(ABLATIONS, nyckel_means, yerr=nyckel_stds, fmt="o", label="Nyckel")
    ax.errorbar(ABLATIONS, hg_means, yerr=hg_stds, fmt="o", label="Huggingface")
    ax.errorbar(ABLATIONS, vertex_means, yerr=vertex_stds, fmt="o", label="Vertex")

    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()"""


def get_latencies():
    for latency_count, service in enumerate(SERVICES):
        latencies = []
        for dataset in DATASETS:
            if os.path.exists(f"data/{dataset}/results"):
                for file in os.listdir(f"data/{dataset}/results"):
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
    _, ax = plt.subplots()
    ax.boxplot(all_latencies, showfliers=False)
    ax.set_xticklabels(SERVICES)
    ax.set_ylabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    get_combined_accuracies()
    get_accuracies()
    get_latencies()
    get_combined_latencies()
