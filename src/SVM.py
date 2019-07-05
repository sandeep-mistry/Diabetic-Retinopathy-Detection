import torch
import torchvision
import torchvision.transforms as transforms
import torchbearer
import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.externals import joblib
from RetDataset import RetDataset

cuda_used = "cuda:0"

train_csv = 'data/trainLabels_balanced_200.csv'
test_csv = 'data/testLabels_balanced_200.csv'
train_images = 'data/train_512_balanced_200/'
test_images = 'data/test_512_balanced_200/'

# train_csv = 'C:/Users/Sandeep/Desktop/Train_data/Train_Labels.csv'
# test_csv = 'C:/Users/Sandeep/Desktop/Test_data/Test_Labels.csv'
# train_images = 'C:/Users/Sandeep/Desktop/Train_data/Train_images'
# test_images = 'C:/Users/Sandeep/Desktop/Test_data/Test_images'

weight_decay = 0.0005  # lambda for L2 reg
epochs = 5
batch_size = 16
feature_extract = False


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Define model and freeze weights
model = models.alexnet(pretrained=True)
# set_parameter_requires_grad(model, feature_extract)
# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, 5)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05),
    transforms.RandomRotation(15),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = RetDataset(train_csv, train_images, transform=transform)
test_set = RetDataset(test_csv, test_images, transform=transform)

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

classes = ('0', '1', '2', '3',
           '4')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if torch.cuda.is_available():
    device = cuda_used
    print("Running with GPU Acceleration")
else:
    print("Running on CPU")
    device = "cpu"

# define the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters())

# Construct a trial object with the model, optimiser and loss.
# Also specify metrics we wish to compute.
trial = torchbearer.Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
# Provide the data to the trial
trial.with_generators(trainloader, test_generator=testloader)

# Run 10 epochs of training
# trial.run(epochs=epochs)
#
# # test the performance
# results = trial.evaluate(data_key=torchbearer.TEST_DATA)
# print(results)

model.load_state_dict(torch.load("weights/weights_balanced_alex_10"))
model.eval()

print('Extracting features for Train Set...')

for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    if device == cuda_used:
        inputs, labels = Variable(inputs.cuda()), \
                         Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # extracting features
    # _, features = model(inputs)
    features = model(inputs)

    if device == cuda_used:
        features = features.cpu()
        labels = labels.cpu()
    feature = features.data.numpy()
    label = labels.data.numpy()
    label = np.reshape(label, (labels.size(0), 1))

    if i == 0:
        featureMatrix = np.copy(feature)
        labelVector = np.copy(label)
    else:
        featureMatrix = np.vstack([featureMatrix, feature])
        labelVector = np.vstack([labelVector, label])

print('Finished feature extraction for Train Set')

print('Extracting features for Test Set...')

for i, data in enumerate(testloader, 0):
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    if device == cuda_used:
        inputs, labels = Variable(inputs.cuda()), \
                         Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # extracting features
    # _, features = model(inputs)
    features = model(inputs)

    if device == cuda_used:
        features = features.cpu()
        labels = labels.cpu()
    feature = features.data.numpy()
    label = labels.data.numpy()
    label = np.reshape(label, (labels.size(0), 1))

    if i == 0:
        featureMatrixTest = np.copy(feature)
        labelVectorTest = np.copy(label)
    else:
        featureMatrixTest = np.vstack([featureMatrixTest, feature])
        labelVectorTest = np.vstack([labelVectorTest, label])

print('Finished feature extraction for Test Set')

# Defining SVM Claasifier
clf = SVC(gamma='auto')
# Train the SVM using Train Set of dataset
clf.fit(featureMatrix, np.ravel(labelVector))

# Test with SVM for Test Set of Dataset
labelVectorPredicted = clf.predict(featureMatrixTest)

labelVectorTest = np.ravel(labelVectorTest)
className = list(classes)
print('GroundTruth', 'Predicted')
print('--------', '--------')
for i in range(10):
    print(className[labelVectorTest[i]], className[labelVectorPredicted[i]])

correct = (labelVectorPredicted == labelVectorTest).sum()
print('Accuracy of the network on test images: %d %%' % (
        100 * correct / labelVectorTest.shape[0]))

class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
c = (labelVectorPredicted == labelVectorTest).squeeze()
for i in range(labelVectorTest.shape[0]):
    label = labelVectorTest[i]
    class_correct[label] += c[i]
    class_total[label] += 1

for i in range(5):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
