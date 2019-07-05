import argparse

import torch
import torchvision
import torchbearer
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader

# For displaying images and numpy operations
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from RetCNN import RetCNN, RetResNet
from RetDataset import RetDataset

# Import Packages for Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier


cuda_used = "cuda:0"


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Perform training')
parser.add_argument('--alexnet', action='store_true', help='Alex net')
parser.add_argument('--vgg', action='store_true', help='VGG 11 BN')
parser.add_argument('--resnet', action='store_true', help='resnet 18')
parser.add_argument('--retresnet', action='store_true', help='resnet 18 with mlp')
parser.add_argument('--inception', action='store_true', help='Inception network')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, help='Learning Rate')
parser.add_argument('--L2', type=float, help='L2 Weight decay')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--ams_grad', action='store_true', help='Use AMSGRAD')
parser.add_argument('--test_images', type=str, help='Test images directory')
parser.add_argument('--train_images', type=str, help='Train images directory')
parser.add_argument('--test_csv', type=str, help='test_csv location')
parser.add_argument('--train_csv', type=str, help='train_csv location')
parser.add_argument('--save_weights_name', type=str, help='Name to save weights under in weights/')
parser.add_argument('--load_weights_name', type=str, help='Name to load weights under in weights/')
parser.add_argument('--cuda', type=int, help='cuda number to choose', default=0)
parser.add_argument('--balanced', action='store_true', help='Is the data set balanced?')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained network?')
parser.add_argument('--freezepretrained', action='store_true', help='freeze the pretrained network?')
parser.add_argument('--transform', type=int, help='Specific transform to use')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

args = parser.parse_args()

classes = ('0', '1', '2', '3',
           '4')


load_weights = False
save_weights = False

if (args.load_weights_name != None):
    load_weights = True
if (args.save_weights_name != None):
    save_weights = True

# use the dataset to get train and test batches.

if args.transform == 1 :
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

elif args.transform == 2:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

train_set = RetDataset(args.train_csv, args.train_images, transform=transform)
test_set = RetDataset(args.test_csv, args.test_images, transform=transform)

trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

if torch.cuda.is_available():
    device = cuda_used
    print("Running with GPU Acceleration")
else:
    print("Running on CPU")
    device = "cpu"

if args.alexnet:
    print("Using Alexnet")
    model = models.alexnet(pretrained=args.pretrained) if args.pretrained else models.alexnet()
    # set_parameter_requires_grad(model, args.freezepretrained)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 5)



elif args.inception:
    print("Using Inception")
    model = models.inception_v3(pretrained=args.pretrained) if args.pretrained else models.alexnet()

elif args.vgg:
    print("Using VGG 11 with BN")
    model = models.vgg11_bn(pretrained=args.pretrained)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 5)

elif args.resnet:
    print("Using Resnet 18")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 5)


elif args.retresnet:
    print("Using Resnet 18 with mlp on the end")
    model = RetResNet()

else:
    print("Using RetCNN")
    model = RetCNN()


model.to(device)

if load_weights:
    print("Loading weights")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.load_weights_name))
    else:
        model.load_state_dict(torch.load(args.load_weights_name, map_location='cpu'))


feature_extractor_model = nn.Sequential(*list(model.children())[:-1])
feature_extractor_model.eval()
feature_extractor_model = feature_extractor_model.to(device)

print('Extracting features for Train Set...')
for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    if device == cuda_used:
        inputs, labels = Variable(inputs.cuda(cuda_used)), \
                         Variable(labels.cuda(cuda_used))
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # extracting features
    # _, features = model(inputs)
    features = feature_extractor_model((inputs).squeeze(3).to(device))

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
        inputs, labels = Variable(inputs.cuda(cuda_used)), \
                         Variable(labels.cuda(cuda_used))
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # extracting features
    # _, features = model(inputs)
    features = feature_extractor_model((inputs).squeeze(3).to(device))


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

nsamples, rsamples, nx, ny = featureMatrix.shape
featureMatrix = featureMatrix.reshape((nsamples,rsamples*nx*ny))

nsamples, rsamples, nx, ny = featureMatrixTest.shape
featureMatrixTest = featureMatrixTest.reshape((nsamples,rsamples*nx*ny))



# Defining Random Forest Claasifier
clf = RandomForestClassifier(n_estimators=1000)

# Train the Random Forest using Train Set of CIFAR-10 Dataset
clf.fit(featureMatrix, np.ravel(labelVector))

# Test with Random Forest for Test Set of CIFAR-10 Dataset
labelVectorPredicted = clf.predict(featureMatrixTest)

labelVectorTest = np.ravel(labelVectorTest)
className = list(classes)
print('GroundTruth', 'Predicted')
print('--------', '--------')
for i in range(10):
    print(className[labelVectorTest[i]], className[labelVectorPredicted[i]])
    
accuracy = accuracy_score(labelVectorTest, labelVectorPredicted)
f1score1 = f1_score(labelVectorTest, labelVectorPredicted, average='macro')
f1score2 = f1_score(labelVectorTest, labelVectorPredicted, average='micro')
f1score3 = f1_score(labelVectorTest, labelVectorPredicted, average='weighted')
f1score4 = f1_score(labelVectorTest, labelVectorPredicted, average=None)
qwk = cohen_kappa_score(labelVectorTest, labelVectorPredicted, weights='quadratic')


print('Accuracy:', accuracy)
print('F1 Score Macro:', f1score1)
print('F1 Score Micro:', f1score2)
print('F1 Score Weighted:', f1score3)
print('F1 Score:', f1score4)
print('QuadWeightedKappa:', qwk)


correct = (labelVectorPredicted == labelVectorTest).sum()
print('Accuracy of the network on the test images: %d %%' % (
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
