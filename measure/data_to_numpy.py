import os
import pickle
import numpy as np

CIFAR10_ROOT = R"/data/gitlab/cifar-10-batches-py"
OUT_FILE = os.path.join(CIFAR10_ROOT, "cifar10.npz")




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

np.savez_compressed(
    OUT_FILE,
    images=images_all,
    labels=labels_all
)

print(f"✔ Gespeichert:  {OUT_FILE}")
print("Shape images:", images_all.shape)
print("Shape labels:", labels_all.shape)


