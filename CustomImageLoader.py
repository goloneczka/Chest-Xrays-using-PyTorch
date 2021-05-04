from torchvision import transforms
from torch.utils.data import Dataset


class CustomImageLoader(Dataset):
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

    def __init__(self, data, name):
        self.data = data
        self.transform = self.train_tfms if name == "train" else self.valid_tfms
        self.arr = [(self.transform(img), label) for img, label in self.data]

    def __getitem__(self, index):
        img, label = self.arr[index]
        return img, label

    def __len__(self):
        return len(self.arr)