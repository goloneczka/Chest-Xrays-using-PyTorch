from math import fabs
from random import randrange
from time import time

import torchvision.transforms.functional as TF
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from CNNModel import CNNModel
import matplotlib.pyplot as plt
import torch.nn.functional as F

DIR = "./data"
IS_MODEL_ALREADY_TRAINED = True


def data_loader(dir):
    print("we are transformating data")
    train_tfms = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                                     transforms.CenterCrop(224),
                                     transforms.RandomRotation(20),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    valid_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    choosenTransformation = train_tfms if '/train' in dir else valid_tfms
    return ImageFolder(dir, transform=choosenTransformation)


def calculate_weights(dataset):
    # we are correctly assuming that in ours datasets are only normal( label 0) and pneumonia(label 1) photos
    # labels_in_dataset = [label for (_, label) in dataset]
    # count_normal_images = len([label for label in labels_in_dataset if label == 0])
    # count_pneumonia_images = len([label for label in labels_in_dataset if label == 1])
    #
    # return [count_normal_images / len(labels_in_dataset), count_pneumonia_images / len(labels_in_dataset)]
    return [1500 / 5200, 3500 / 5200]  # -> for faster calculation


def balance_data_to_learn(dataset):
    print("we are balacing data")
    weights = calculate_weights(dataset)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 100, replacement=True)
    return torch.utils.data.DataLoader(dataset, batch_size=28, sampler=sampler, num_workers=1, pin_memory=False)


def loss_plot(avg_loses):
    plt.plot(avg_loses["avg_train_loses"], marker='o', color="blue", label="treningowe")
    plt.plot(avg_loses["avg_valid_loses"], marker='o', color='olive', label="walidacyjne")
    plt.xlabel("epoka")
    plt.ylabel("bład")
    plt.legend()
    plt.savefig('loss.png')


def learning(train_dataset, validation_dataset, model):
    print("we are learning model")

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003,
                                momentum=0.9)  # used in first exp
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.03, momentum=0.7)     # used in second exp
    criterion = torch.nn.CrossEntropyLoss()

    time0 = time()
    avg_prev_loss = -1
    avg_loses = {"avg_train_loses": [], "avg_valid_loses": []}

    for e in range(25):
        train_loss = 0
        for images, labels in train_dataset:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_loses["avg_train_loses"].append(train_loss / len(train_dataset))
        print("Epoch {} - Training loss: {}".format(e, avg_loses["avg_train_loses"][-1]))

        valid_loss = 0
        for images, labels in validation_dataset:
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item()
        avg_loses["avg_valid_loses"].append(valid_loss / len(train_dataset))
        print("Epoch {} - Valid loss: {}".format(e, avg_loses["avg_valid_loses"][-1]))

        if fabs(avg_loses["avg_train_loses"][-1] - avg_prev_loss) < 0.001:
            break
        avg_prev_loss = avg_loses["avg_valid_loses"][-1]

    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    loss_plot(avg_loses)
    return model


def get_learned_model(train_dataset, validation_dataset):
    print("we are getting model")
    if IS_MODEL_ALREADY_TRAINED:
        model = torch.load("./CNNModel.pt")
        model.eval()
        print('Model from file')
        return model

    model = learning(train_dataset, validation_dataset, CNNModel())
    torch.save(model, "./CNNModel.pt")
    print('Model saved to file')
    return model


def save_wrong_pred(pred, images, labels):
    wrong_idx = (pred != labels.view_as(pred)).nonzero()[:, 0]
    print(wrong_idx)
    if not len(wrong_idx):
        return

    sample = images[wrong_idx][-1]
    wrong_pred = pred[wrong_idx][-1]
    actual_pred = labels.view_as(pred)[wrong_idx][-1]

    img = TF.to_pil_image(sample)
    img.save('./wrong_pred/pred{}_actual{}.png'.format(
        wrong_pred.item(), actual_pred.item()))


def check_model(model, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pred_arr, true_arr = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

            pred_arr.extend(pred)
            true_arr.extend(labels.numpy())

            save_wrong_pred(pred, images, labels)

    print(correct, ' ', len(val_loader.dataset), ' ', test_loss)
    print("\nModel Accuracy =", (100 * correct / len(val_loader.dataset)))
    return pred_arr, true_arr


if __name__ == "__main__":
    train_dataset = data_loader(DIR + '/train')
    validation_dataset = data_loader(DIR + '/val')
    test_dataset = data_loader(DIR + '/test')

    train_dataset = balance_data_to_learn(train_dataset)
    validation_dataset = torch.utils.data.DataLoader(validation_dataset, batch_size=28, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=28, shuffle=True)

    model = get_learned_model(train_dataset, validation_dataset)
    pred_arr, true_arr = check_model(model, train_dataset)
