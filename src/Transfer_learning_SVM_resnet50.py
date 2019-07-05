"""

This file is intended to replicate the transfer learning method employed by Jon Hare in the 6_1 Deep Learning lab.
I had issues with getting my GPU to work and ran out of RAM when attempting to use my CPU.

"""


from torchvision.models import resnet50
from torchvision.models import alexnet
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from RetDataset import RetDataset
import numpy as np
from sklearn.svm import SVC

# device = "cuda:3" if torch.cuda.is_available() else "cpu"
# print(device)

# device = "cuda:0"
if torch.cuda.is_available():
    device = "cuda:0"
    print("Running with GPU Acceleration")
else:
    device = "cpu"
    print("Running on CPU")


#Use 'resnet50' model
model = resnet50(pretrained=True)
# model = alexnet(pretrained=True)
feature_extractor_model = nn.Sequential(*list(model.children())[:-2], nn.AdaptiveAvgPool2d((1,1)))
# feature_extractor_model = nn.Sequential(*list(model.classifier.children())[:-1])
feature_extractor_model.eval()
feature_extractor_model = feature_extractor_model.to(device)

batch_size = 2
transform = transforms.Compose([
    transforms.Resize((240, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

train_csv = 'data/trainLabels.csv'
test_csv = 'data/testLabels.csv'
train_images = 'data/train_512/'
test_images = 'data/test_512/'

train_set = RetDataset(train_csv, train_images, transform=transform)
test_set = RetDataset(test_csv, test_images, transform=transform)

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Extract training features
training_features = []
training_labels = []
for imgs, labels in trainloader:
    print(".")
    features = feature_extractor_model(imgs.to(device))
    for i in range(features.shape[0]):
        training_features.append(features[i].reshape(-1).cpu().detach().numpy())
        training_labels.append(labels[i])

training_features = np.array(training_features)
training_labels = np.array(training_labels)
# np.save("training_features.npy", training_features)
# np.save("training_labels.npy", training_labels)

# Extract testing features
testing_features = []
testing_labels = []
for imgs, labels in testloader:
    print(".")
    features = feature_extractor_model(imgs.to(device))
    for i in range(features.shape[0]):
        testing_features.append(features[i].reshape(-1).cpu().detach().numpy())
        testing_labels.append(labels[i])
testing_features = np.array(testing_features)
testing_labels = np.array(testing_labels)
# np.save("testing_features.npy", testing_features)
# np.save("testing_labels.npy", testing_labels)

# training_features = np.load('training_features.npy')
# training_labels = np.load('training_labels.npy')

# testing_features = np.load('testing_features.npy')
# testing_labels = np.load('testing_labels.npy')

# Use SVM model
model = SVC(gamma='scale')
model.fit(training_features, training_labels)
print(model.score(testing_features, testing_labels))