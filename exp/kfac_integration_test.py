import torch
from torch.nn import CrossEntropyLoss

from exp.loading.load_mnist import MNISTLoader
from exp.models.chen2018 import mnist_model
from exp.third_party.optimizers.ekfac import EKFACOptimizer
from exp.third_party.optimizers.kfac import KFACOptimizer

# global hyperparameters
batch = 500
num_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dirname = 'kfac_test/mnist'
# data_dir = directory_in_data(dirname)
# logs_per_epoch = 5

USE_OPTIMIZER = "ekfac"


def main():
    model = mnist_model().to(device)
    loss_function = CrossEntropyLoss()
    data_loader = MNISTLoader(train_batch_size=batch, test_batch_size=batch)

    if USE_OPTIMIZER == "kfac":
        optimizer = KFACOptimizer(
            model,
            lr=0.001,
            momentum=0.9,
            stat_decay=0.95,
            damping=0.001,
            kl_clip=0.001,
            weight_decay=0,
            TCov=10,
            TInv=100,
            batch_averaged=True,
        )
    elif USE_OPTIMIZER == "ekfac":
        optimizer = EKFACOptimizer(
            model,
            lr=0.001,
            momentum=0.9,
            stat_decay=0.95,
            damping=0.001,
            kl_clip=0.001,
            weight_decay=0,
            TCov=10,
            TScal=10,
            TInv=100,
            batch_averaged=True,
        )
    else:
        raise ValueError("Unknown optimizer")

    # log some metrics
    train_epoch = []
    batch_loss = []
    batch_acc = []

    samples = 0
    samples_per_epoch = 60000.0

    train_loader = data_loader.train_loader()

    for epoch in range(num_epochs):
        iters = len(train_loader)

        for i, (images, labels) in enumerate(train_loader):
            # reshape and load to device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute statistics
            total = labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / total

            # update lists
            samples += total
            train_epoch.append(samples / samples_per_epoch)
            batch_loss.append(loss.item())
            batch_acc.append(accuracy)

            # print every 5 iterations
            if i % 5 == 0:
                print(
                    "Epoch [{}/{}], Iter. [{}/{}], Loss: {:.4f}, Acc.: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, iters, loss.item(), accuracy
                    )
                )


if __name__ == "__main__":
    main()
