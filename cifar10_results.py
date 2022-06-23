import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import CenteredNorm

show = False  # True to print the whole dictionary, otherwise only aggregated results will be displayed

with open("results/cifar10_timeit_iterate.pickle", "rb") as h:
    times_iterate = pickle.load(h)
    if show:
        print(json.dumps(times_iterate, indent=4))

train_ffcv_cpu = [
    x["train"] / x["epochs"] for x in times_iterate["ffcv"] if x["device"] == "cpu"
]
train_torch_cpu = [
    x["train"] / x["epochs"] for x in times_iterate["torch"] if x["device"] == "cpu"
]
train_ffcv_cuda = [
    x["train"] / x["epochs"] for x in times_iterate["ffcv"] if x["device"] == "cuda"
]

count_cuda = 0
count_torch = 0
ratio_cuda = []
ratio_torch = []
for x in times_iterate["torch"]:
    if x["device"] == "cuda":
        ffcv_cuda = [
            y
            for y in times_iterate["ffcv"]
            if y["device"] == "cuda"
            and y["epochs"] == x["epochs"]
            and y["workers"] == x["workers"]
        ][0]["train"]
        torch_cpu = [
            y
            for y in times_iterate["torch"]
            if y["device"] == "cpu"
            and y["epochs"] == x["epochs"]
            and y["workers"] == x["workers"]
        ][0]["train"]
        torch_cuda = x["train"]

        count_cuda += torch_cuda > ffcv_cuda
        count_torch += torch_cuda > torch_cpu
        ratio_cuda.append(torch_cuda / ffcv_cuda)
        ratio_torch.append(torch_cpu / torch_cuda)

count_cpu = sum(
    [train_torch_cpu[i] > train_ffcv_cpu[i] for i in range(len(train_torch_cpu))]
)
count_mix = sum(
    [train_torch_cpu[i] > train_ffcv_cuda[i] for i in range(len(train_torch_cpu))]
)
count_ffcv = sum(
    [train_ffcv_cpu[i] < train_ffcv_cuda[i] for i in range(len(train_ffcv_cpu))]
)
n = len(train_torch_cpu)
n_torch_cuda = len([x for x in times_iterate["torch"] if x["device"] == "cuda"])
print("|===== Comparison among frameworks and devices =====|")
print(
    f"[loading time, cpu] fraction of cases torch > ffcv: {count_cpu}/{n}={count_cpu / n * 100:.1f}%\n"
    f"[loading time, cuda] fraction of cases torch > ffcv: {count_cuda}/{n_torch_cuda}={count_cuda / n_torch_cuda * 100:.1f}%\n"
    f"[loading time, mix] fraction of cases torch > ffcv: {count_mix}/{n}={count_mix / n * 100:.1f}%\n"
    f"[loading time, ffcv] fraction of cases ffcv_cuda > ffcv_cpu: {count_ffcv}/{n}={count_ffcv / n * 100:.1f}%\n"
    f"[loading time, torch] fraction of cases torch_cuda > torch_cpu: {count_torch}/{n_torch_cuda}={count_torch / n_torch_cuda * 100:.1f}%\n"
)

ratio_cpu = [
    train_torch_cpu[i] / train_ffcv_cpu[i] for i in range(len(train_torch_cpu))
]
ratio_mix = [
    train_torch_cpu[i] / train_ffcv_cuda[i] for i in range(len(train_torch_cpu))
]
ratio_ffcv = [
    train_ffcv_cpu[i] / train_ffcv_cuda[i] for i in range(len(train_ffcv_cpu))
]
print(
    f"[loading time, cpu] ratio torch/ffcv: mean {np.mean(ratio_cpu):.1f}, std {np.std(ratio_cpu):.1f}, min {min(ratio_cpu):.1f}, max {max(ratio_cpu):.1f}\n"
    f"[loading time, cuda] ratio torch/ffcv: mean {np.mean(ratio_cuda):.1f}, std {np.std(ratio_cuda):.1f}, min {min(ratio_cuda):.1f}, max {max(ratio_cuda):.1f}\n"
    f"[loading time, mix] ratio torch/ffcv: mean {np.mean(ratio_mix):.1f}, std {np.std(ratio_mix):.1f}, min {min(ratio_mix):.1f}, max {max(ratio_mix):.1f}\n"
    f"[loading time, ffcv] ratio ffcv_cpu/ffcv_cuda: mean {np.mean(ratio_ffcv):.1f}, std {np.std(ratio_ffcv):.1f}, min {min(ratio_ffcv):.1f}, max {max(ratio_ffcv):.1f}\n"
    f"[loading time, torch] ratio torch_cpu/torch_cuda: mean {np.mean(ratio_torch):.5f}, std {np.std(ratio_torch):.5f}, min {min(ratio_torch):.5f}, max {max(ratio_torch):.5f}"
)

print("\n\n|===== Best num_workers per framework, device and number of epochs =====|")
print("[ffcv,cpu]")
for epochs in [1, 10, 50, 100]:
    best = min(
        [
            x["train"]
            for x in times_iterate["ffcv"]
            if x["device"] == "cpu" and x["epochs"] == epochs
        ]
    )
    print(
        [
            x
            for x in times_iterate["ffcv"]
            if x["device"] == "cpu" and x["epochs"] == epochs and x["train"] == best
        ]
    )

print("\n[torch,cpu]")
for epochs in [1, 10, 50, 100]:
    best = min(
        [
            x["train"]
            for x in times_iterate["torch"]
            if x["device"] == "cpu" and x["epochs"] == epochs
        ]
    )
    print(
        [
            x
            for x in times_iterate["torch"]
            if x["device"] == "cpu" and x["epochs"] == epochs and x["train"] == best
        ]
    )

print("\n[ffcv,cuda]")
for epochs in [1, 10, 50, 100]:
    best = min(
        [
            x["train"]
            for x in times_iterate["ffcv"]
            if x["device"] == "cuda" and x["epochs"] == epochs
        ]
    )
    print(
        [
            x
            for x in times_iterate["ffcv"]
            if x["device"] == "cuda" and x["epochs"] == epochs and x["train"] == best
        ]
    )

print("\n[torch,cuda]")
for epochs in [1, 10, 50, 100]:
    best = min(
        [
            x["train"]
            for x in times_iterate["torch"]
            if x["device"] == "cuda" and x["epochs"] == epochs
        ]
    )
    print(
        [
            x
            for x in times_iterate["torch"]
            if x["device"] == "cuda" and x["epochs"] == epochs and x["train"] == best
        ]
    )

ffcv = {"cuda": [], "cpu": []}
for epochs in [1, 10, 50, 100]:
    ffcv["cuda"].append(
        [
            x["train"] / x["epochs"]
            for x in times_iterate["ffcv"]
            if x["epochs"] == epochs and x["device"] == "cuda"
        ]
    )
for epochs in [1, 10, 50, 100]:
    ffcv["cpu"].append(
        [
            x["train"] / x["epochs"]
            for x in times_iterate["ffcv"]
            if x["epochs"] == epochs and x["device"] == "cpu"
        ]
    )

torch = {"cuda": [], "cpu": []}
for epochs in [1, 10, 50, 100]:
    torch["cuda"].append(
        [
            x["train"] / x["epochs"]
            for x in times_iterate["torch"]
            if x["epochs"] == epochs and x["device"] == "cuda"
        ]
        + [np.nan] * 8
    )
for epochs in [1, 10, 50, 100]:
    torch["cpu"].append(
        [
            x["train"] / x["epochs"]
            for x in times_iterate["torch"]
            if x["epochs"] == epochs and x["device"] == "cpu"
        ]
    )

plt.figure(figsize=(9, 6))
plt.suptitle("per epoch dataloading time difference", fontsize=15)
c = 1
for i in torch:
    for j in ffcv:
        plt.subplot(2, 2, c)
        diff = np.array(torch[i]) - np.array(ffcv[j])
        cmap = cm.coolwarm
        plt.imshow(diff, interpolation="none", norm=CenteredNorm(), cmap=cmap)
        plt.colorbar()
        plt.yticks(np.arange(4), [1, 10, 50, 100])
        plt.xlabel("num_workers")
        plt.ylabel("epochs")
        plt.title(f"torch[{i}] - ffcv[{j}]")
        c += 1
plt.savefig("results/cifar10_loading_diff.jpg")

plt.figure(figsize=(9, 6))
center = np.arange(9)
width = 0.1

plt.subplot(2, 2, 1)
plt.bar(center - 1.5 * width, ffcv["cuda"][0], width=width, label="1 ep.")
plt.bar(center - 0.5 * width, ffcv["cuda"][1], width=width, label="10 ep.")
plt.bar(center + 0.5 * width, ffcv["cuda"][2], width=width, label="50 ep.")
plt.bar(center + 1.5 * width, ffcv["cuda"][3], width=width, label="100 ep.")
plt.xticks([])
plt.legend()
plt.ylabel("loading time/epoch")
plt.title("ffcv[cuda]")

plt.subplot(2, 2, 2)
plt.bar(center - 1.5 * width, ffcv["cpu"][0], width=width, label="1 ep.")
plt.bar(center - 0.5 * width, ffcv["cpu"][1], width=width, label="10 ep.")
plt.bar(center + 0.5 * width, ffcv["cpu"][2], width=width, label="50 ep.")
plt.bar(center + 1.5 * width, ffcv["cpu"][3], width=width, label="100 ep.")
plt.xticks([])
plt.legend()
plt.ylabel("loading time/epoch")
plt.title("ffcv[cpu]")

plt.subplot(2, 2, 3)
plt.bar(center - 1.5 * width, torch["cuda"][0], width=width, label="1 ep.")
plt.bar(center - 0.5 * width, torch["cuda"][1], width=width, label="10 ep.")
plt.bar(center + 0.5 * width, torch["cuda"][2], width=width, label="50 ep.")
plt.bar(center + 1.5 * width, torch["cuda"][3], width=width, label="100 ep.")
plt.xticks(center)
plt.legend()
plt.xlabel("num_workers")
plt.ylabel("loading time/epoch")
plt.title("torch[cuda]")

plt.subplot(2, 2, 4)
plt.bar(center - 1.5 * width, torch["cpu"][0], width=width, label="1 ep.")
plt.bar(center - 0.5 * width, torch["cpu"][1], width=width, label="10 ep.")
plt.bar(center + 0.5 * width, torch["cpu"][2], width=width, label="50 ep.")
plt.bar(center + 1.5 * width, torch["cpu"][3], width=width, label="100 ep.")
plt.xticks(center)
plt.legend()
plt.xlabel("num_workers")
plt.ylabel("loading time/epoch")
plt.title("torch[cpu]")

plt.savefig("results/cifar10_loading.jpg")

# training
with open("results/cifar10_timeit_train.pickle", "rb") as h:
    times_train = pickle.load(h)
    print("\nTraining experiment:")
    print(json.dumps(times_train, indent=4))

plt.figure()
center = np.arange(2)
width = 0.25
plt.bar(
    center - 0.5 * width,
    [times_train["ffcv"][0]["train"], times_train["ffcv"][1]["train"]],
    width=width,
    label="ffcv[cuda]",
)
plt.bar(
    center + 0.5 * width,
    [times_train["torch"][0]["train"], times_train["torch"][1]["train"]],
    width=width,
    label="torch[cpu]",
)
plt.legend()
plt.xticks(center, [0, 8])
plt.xlabel("num_workers")
plt.ylabel("training time")

plt.savefig("results/cifar10_training.jpg")
