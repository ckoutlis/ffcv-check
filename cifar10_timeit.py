import os.path
from typing import List
import multiprocessing as mp
import time
import pickle
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    RandomHorizontalFlip,
    RandomTranslate,
    Convert,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze


def make_dataloaders_torch(
    train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, device=None
):
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)

    loaders = {}

    loaders["train"] = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=train_dataset,
            train=True,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(2 / 32, 2 / 32),
                        fill=tuple(map(int, CIFAR_MEAN)),
                    ),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.to(device)),
                    transforms.ConvertImageDtype(torch.float16),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory is set to True in all cases but the case of torch dataloader
        # on cuda device and num_workers=0. Otherwise it raises and error
        # in that particular case. Same in "test" loader.
        pin_memory=not (device == "cuda" and num_workers == 0),
        drop_last=True,
    )

    loaders["test"] = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=val_dataset,
            train=False,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.to(device)),
                    transforms.ConvertImageDtype(torch.float16),
                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not (device == "cuda" and num_workers == 0),
        drop_last=False,
    )

    return loaders


def make_dataloaders_ffcv(
    train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, device=None
):
    paths = {"train": train_dataset, "test": val_dataset}

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ["train", "test"]:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(device),
            Squeeze(),
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == "train":
            image_pipeline.extend(
                [
                    RandomHorizontalFlip(),
                    RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                ]
            )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice(device, non_blocking=True),
                ToTorchImage(),
                Convert(torch.float16),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        ordering = OrderOption.RANDOM if name == "train" else OrderOption.SEQUENTIAL

        loaders[name] = Loader(
            paths[name],
            batch_size=batch_size,
            num_workers=num_workers,
            order=ordering,
            drop_last=(name == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    return loaders


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        torch.nn.BatchNorm2d(channels_out),
        torch.nn.ReLU(inplace=True),
    )


def construct_model():
    num_class = 10
    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        Mul(0.2),
    )
    model = model.to(memory_format=torch.channels_last).cuda()
    return model


def train(
    model,
    loaders,
    device,
    lr=0.5,
    epochs=24,
    label_smoothing=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    lr_peak_epoch=5,
):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders["train"])

    lr_schedule = np.interp(
        np.arange((epochs + 1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0],
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in range(epochs):
        for ims, labs in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims if device == "cuda" else ims.to("cuda"))
                loss = loss_fn(out, labs if device == "cuda" else labs.to("cuda"))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()


def evaluate(model, loaders, device):
    model.eval()
    with torch.no_grad():
        total_correct, total_num = 0.0, 0.0
        for ims, labs in loaders["test"]:
            if device == "cpu":
                ims = ims.to("cuda")
                labs = labs.to("cuda")
            with autocast():
                out = model(ims)
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]
    return total_correct / total_num


def iterate(loaders, mode, epochs):
    for _ in range(epochs):
        for _ in loaders[mode]:
            pass


def timeit(modes, EPOCHS, WORKERS, train_bool, fn):
    if os.path.exists(f"results/{fn}.pickle"):
        with open(f"results/{fn}.pickle", "rb") as h:
            times = pickle.load(h)
    else:
        times = {}

    counter = 0
    total = len(EPOCHS) * len(WORKERS) * sum([len(modes[m]) for m in modes])
    for mode in modes:
        if mode not in times:
            times[mode] = []

        for epochs in EPOCHS:
            for workers in WORKERS:
                for device in modes[mode]:
                    counter += 1

                    if any(
                        [
                            x["epochs"] == epochs
                            and x["workers"] == workers
                            and x["device"] == device
                            for x in times[mode]
                        ]
                    ):
                        print(
                            f"[{'train' if train_bool else 'iterate'},{counter}/{total}] mode: {mode}, epochs: {epochs}, workers: {workers}, device: {device} --- ALREADY CONDUCTED"
                        )
                        continue
                    elif mode == "torch" and device == "cuda" and workers > 0:
                        # Continue when the setting has already run or when
                        # the setting involves standard torch dataloader
                        # on cuda device and num_workers>0 because this
                        # scenario is not supported by pytorch.
                        # It is claimed that the latter might work using 'spawn', namely
                        # torch.multiprocessing.set_start_method('spawn', force=True)
                        # but I tested it and it does not work.
                        print(
                            f"[{'train' if train_bool else 'iterate'},{counter}/{total}] mode: {mode}, epochs: {epochs}, workers: {workers}, device: {device} --- NOT APPLICABLE"
                        )
                        continue

                    t0 = time.time()
                    if mode == "ffcv":
                        loaders = make_dataloaders_ffcv(
                            train_dataset="./data/cifar_train.beton",
                            val_dataset="./data/cifar_test.beton",
                            batch_size=512,
                            num_workers=workers,
                            device=device,
                        )
                    elif mode == "torch":
                        loaders = make_dataloaders_torch(
                            train_dataset="./data",
                            val_dataset="./data",
                            batch_size=512,
                            num_workers=workers,
                            device=device,
                        )

                    t1 = time.time()
                    if train_bool:
                        model = construct_model()
                        train(
                            model=model, loaders=loaders, device=device, epochs=epochs
                        )
                    else:
                        iterate(loaders=loaders, mode="train", epochs=epochs)

                    t2 = time.time()
                    if train_bool:
                        accuracy = evaluate(model=model, loaders=loaders, device=device)
                    else:
                        iterate(loaders=loaders, mode="test", epochs=epochs)
                        accuracy = "not applicable"

                    t3 = time.time()

                    times[mode].append(
                        {
                            "epochs": epochs,
                            "device": device,
                            "workers": workers,
                            "train": t2 - t1,
                            "test": t3 - t2,
                            "make_dataloaders": t1 - t0,
                            "train_bool": train_bool,
                            "accuracy": accuracy,
                        }
                    )
                    with open(f"results/{fn}.pickle", "wb") as h:
                        pickle.dump(times, h, protocol=pickle.HIGHEST_PROTOCOL)

                    print(
                        f"[{'train' if train_bool else 'iterate'},{counter}/{total}] mode: {mode}, epochs: {epochs}, workers: {workers}, device: {device}"
                    )


if __name__ == "__main__":
    exps_start = time.time()

    # Experiments on loading the data
    EPOCHS = [1, 10, 50, 100]
    WORKERS = range(mp.cpu_count() + 1)
    timeit(
        modes={"ffcv": ["cpu", "cuda"], "torch": ["cpu", "cuda"]},
        EPOCHS=EPOCHS,
        WORKERS=WORKERS,
        train_bool=False,
        fn="cifar10_timeit_iterate",
    )

    # Experiments on loading the data and training a small neural network
    EPOCHS = [24]
    WORKERS = [0, 8]
    timeit(
        modes={"ffcv": ["cuda"], "torch": ["cpu"]},
        EPOCHS=EPOCHS,
        WORKERS=WORKERS,
        train_bool=True,
        fn="cifar10_timeit_train",
    )

    exps_end = time.time()

    print(f"Total time: {exps_end - exps_start:.1f} sec")
    # Total time: ???
