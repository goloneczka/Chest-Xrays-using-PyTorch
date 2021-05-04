import shutil
from random import randrange
from time import time

import numpy as np
import torchvision.transforms.functional as TF
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from CNNModel import CNNModel
import matplotlib.pyplot as plt

from CustomImageLoader import CustomImageLoader

DIR = "./data"
IS_MODEL_ALREADY_TRAINED = False


def split_data_to_valid_and_train(train_dataset):
    print("Spliting data into valid / train dataset")
    torch.manual_seed(42)
    val_size = int(len(train_dataset) * 0.15)

    return random_split(train_dataset, [len(train_dataset) - val_size, val_size])


def data_loader(dir):
    print("we are transformating data")
    train_tfms = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                                     transforms.CenterCrop(224),
                                     transforms.RandomRotation(20),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    choosenTransformation = train_tfms if '/train' or '/val' in dir else test_tfms
    return ImageFolder(dir, transform=choosenTransformation)


def calculate_weights(dataset):
    # we are correctly assuming that in ours datasets are only normal( label 0) and pneumonia(label 1) photos
    labels_in_dataset = [label for (_, label) in dataset]
    count_normal_images = len([label for label in labels_in_dataset if label == 0])
    print(count_normal_images, ' ', len(labels_in_dataset))
    weight = [1 / count_normal_images, 1 / (len(labels_in_dataset) - count_normal_images)]
    return np.array([weight[t] for t in labels_in_dataset])


def balance_data_to_learn(dataset):
    print("we are balacing data")
    weights = calculate_weights(dataset)
    samples_weight = torch.from_numpy(weights)
    samples_weight = samples_weight.double()

    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    return torch.utils.data.DataLoader(dataset, batch_size=64, sampler=sampler)


def loss_plot(avg_loses):
    plt.plot(avg_loses["avg_train_loses"], marker='o', color="blue", label="treningowe")
    plt.plot(avg_loses["avg_valid_loses"], marker='o', color='olive', label="walidacyjne")
    plt.xlabel("epoka")
    plt.ylabel("bład")
    plt.legend()
    plt.savefig('loss.png')


def stop_criterium(avg_valid_loses, model):
    if len(avg_valid_loses) <= 3:
        torch.save(model, './saved_models/{}CNNModel.pt'.format(len(avg_valid_loses)))
        return False
    shutil.move("./saved_models/2CNNModel.pt", "./saved_models/1CNNModel.pt")
    shutil.move("./saved_models/3CNNModel.pt", "./saved_models/2CNNModel.pt")
    torch.save(model, "./saved_models/3CNNModel.pt")
    return avg_valid_loses[-3] < avg_valid_loses[-2] < avg_valid_loses[-1]


def learning(train_dataset, validation_dataset, model):
    print("we are learning model")

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003,
                                momentum=0.9)  # used in first exp
    criterion = torch.nn.CrossEntropyLoss()

    time0 = time()
    avg_loses = {"avg_train_loses": [], "avg_valid_loses": []}

    for e in range(20):
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
        avg_loses["avg_valid_loses"].append(valid_loss / len(validation_dataset))
        print("Epoch {} - Valid loss: {}".format(e, avg_loses["avg_valid_loses"][-1]))

        if stop_criterium(avg_loses["avg_valid_loses"], model):
            model = torch.load("./saved_models/1CNNModel.pt")
            model.eval()
            break

    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    loss_plot(avg_loses)
    return model


def get_learned_model(train_dataset = None, validation_dataset = None):
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
    if not len(wrong_idx):
        return

    sample = images[wrong_idx][-1]
    wrong_pred = pred[wrong_idx][-1]
    actual_pred = labels.view_as(pred)[wrong_idx][-1]

    img = TF.to_pil_image(sample)
    img.save('./wrong_pred/{}id_pred{}_actual{}.png'.format(
        randrange(1000), wrong_pred.item(), actual_pred.item()))


def check_model(model, val_loader):
    model.eval()
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

    print(correct, ' ', len(val_loader.dataset), ' ')
    print("\nModel Accuracy =", (100 * correct / len(val_loader.dataset)))
    return pred_arr, true_arr


def confusion_matrix(pred_list, true_list):
    stack = torch.stack((
        torch.tensor(pred_list),
        torch.tensor(true_list)
    ), dim=1)

    cmt = torch.zeros(2, 2, dtype=torch.int64)

    for p in stack:
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1

    return cmt


if __name__ == "__main__":

    if not IS_MODEL_ALREADY_TRAINED:
        train_dataset, validate_dataset = split_data_to_valid_and_train(ImageFolder(DIR + "/train"))
        validate_dataset = CustomImageLoader(validate_dataset + ImageFolder(DIR + "/val"), " ")
        train_dataset = CustomImageLoader(train_dataset, "train")

    test_dataset = CustomImageLoader(ImageFolder(DIR + "/test"), " ")

    if not IS_MODEL_ALREADY_TRAINED:
        train_dataset = balance_data_to_learn(train_dataset)
        validate_dataset = torch.utils.data.DataLoader(validate_dataset, batch_size=64, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    if not IS_MODEL_ALREADY_TRAINED:
        model = get_learned_model(train_dataset, validate_dataset)
    else:
        model = get_learned_model()

    pred_arr, true_arr = check_model(model, test_dataset)
    print("matrix: ", confusion_matrix(pred_arr, true_arr))

