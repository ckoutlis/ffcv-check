import torchvision
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField

train_dataset, val_dataset = "./data/cifar_train.beton", "./data/cifar_test.beton"

datasets = {
    "train": torchvision.datasets.CIFAR10("./data", train=True, download=True),
    "test": torchvision.datasets.CIFAR10("./data", train=False, download=True),
}

for (name, ds) in datasets.items():
    path = train_dataset if name == "train" else val_dataset
    writer = DatasetWriter(path, {"image": RGBImageField(), "label": IntField()})
    writer.from_indexed_dataset(ds)
