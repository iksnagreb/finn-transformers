import os
import pickle
import numpy as np
from torchvision import transforms
import torch

CIFAR10_ROOT = R"/data/gitlab/cifar-10-batches-py"
OUT_FILE = os.path.join(CIFAR10_ROOT, "cifar10.npz")

# Transformation to be applied to the input images: Rather basic
# preprocessing turning images into tensors and normalizing with only
# minimal data augmentation
tf = transforms.Compose([
    # Convert from PIL image to PyTorch tensors
    transforms.ToTensor(),
    # Random horizontal flip in 50% of the cases
    transforms.RandomHorizontalFlip(),
    # CIFAR-10 statistics on the whole training set
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])



def load_batch(path):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    images = batch[b"data"].reshape(-1, 3, 32, 32)
    labels = np.array(batch[b"labels"])
    return images, labels

images_all = []
labels_all = []

# data_batch_1 ... data_batch_5
for i in range(1, 6):
    imgs, lbls = load_batch(os.path.join(CIFAR10_ROOT, f"data_batch_{i}"))
    images_all.append(imgs)
    labels_all.append(lbls)

# test_batch
imgs, lbls = load_batch(os.path.join(CIFAR10_ROOT, "test_batch"))
images_all.append(imgs)
labels_all.append(lbls)

images_all = np.concatenate(images_all)
labels_all = np.concatenate(labels_all)

images_all_transformed = []

for i in range(images_all.shape[0]):
    img = images_all[i].transpose(1, 2, 0)  # CIFAR Pickle ist (C,H,W)
    img = tf(img)  # -> tensor float32 [0,1] + normalize
    images_all_transformed.append(img.numpy())

images_all_transformed = np.stack(images_all_transformed)


np.savez_compressed(
    OUT_FILE,
    images=images_all_transformed,
    labels=labels_all
)

print(f"✔ Gespeichert:  {OUT_FILE}")
print("Shape images:", images_all.shape)
print("Shape labels:", labels_all.shape)


