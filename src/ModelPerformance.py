import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
from numpy.random.mtrand import randint
from sklearn.decomposition import PCA
from torch import nn
from sklearn.metrics import classification_report
from RetCNN import RetCNN
from RetDataset import RetDataset

load_weights = True
load_weights_dir = "weights/0706unfrozenresnet"
retcnn = False
alexnet = False
vgg = False
resnet = True
csv_file = 'data/testLabels.csv'
image_root = 'data/test_512'


def accuary_metrics(confusion_matrix):
    acc_denom = 0
    acc = 0
    se = 0
    se_denom = 0
    sp = 0
    sp_denom = 0

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            acc_denom += confusion_matrix[i][j]
            if i == j:
                acc += confusion_matrix[i][j]
                if i > 0:
                    se += confusion_matrix[i][j]
            if i > 0 and j == 0:
                se_denom += confusion_matrix[i][j]
            if i == 0 and j == 0:
                sp = confusion_matrix[i][j]
            if i == 0 and j > 0:
                sp_denom += confusion_matrix[i][j]

    if acc_denom == 0:
        acc_denom = 1
    if se_denom == 0:
        se_denom = 1
    if sp_denom == 0:
        sp_denom = 1

    acc = acc / acc_denom
    se = se / se_denom
    sp = sp / sp_denom

    return np.array([acc, se, sp])


# use the dataset to get train and test batches.
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.2),
#     transforms.RandomRotation(15),
#     transforms.ToTensor(),  # convert to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05),
    transforms.RandomRotation(15),
    transforms.CenterCrop(400),
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


data = RetDataset(csv_file, image_root, transform=transform)

if alexnet:
    model = models.alexnet(pretrained=True)
    set_parameter_requires_grad(model, True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 5)

    feature_extractor = models.alexnet(pretrained=True)
    # feature_extractor.load_state_dict(torch.load(load_weights_dir, map_location='cpu')) if not frozen
    beforelabel = nn.Sequential(*list(model.classifier.children())[:-1])
    feature_extractor.classifier = beforelabel

elif vgg:
    model = models.vgg11_bn(pretrained=True)
    set_parameter_requires_grad(model, True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 5)

    feature_extractor = models.vgg11_bn(pretrained=True)
    beforelabel = nn.Sequential(*list(model.classifier.children())[:-2])
    feature_extractor.classifier = beforelabel

elif resnet:
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 5)

    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    print(feature_extractor)

else:
    model = RetCNN()
    feature_extractor = RetCNN(True)

if torch.cuda.is_available():
    device = "cuda"
    if load_weights: model.load_state_dict(torch.load(load_weights_dir))
    if retcnn: feature_extractor.load_state_dict(torch.load(load_weights_dir))
    model.cuda()
else:
    device = "cpu"
    if load_weights: model.load_state_dict(torch.load(load_weights_dir, map_location='cpu'))
    if retcnn: feature_extractor.load_state_dict(torch.load(load_weights_dir, map_location='cpu'))

# find different way of creating this else it will take up too much memory, this is garbage code for garbage people
# feature_extractor_model = nn.Sequential(*list(model.children())[:-5])
model.train(False)
model.eval()

# take a bunch of random indexes, read the data
feature_vectors = []
predictions = []
targets = []
cm = np.zeros((5, 5))  # to store the confusion matrix

for i in range(5):
    for _ in range(25):
        idx = randint(0, data.__len__())

        while data.__getitem__(idx)[1] != i:
            idx = randint(0, data.__len__())

        input = torch.Tensor([data.__getitem__(idx)[0].tolist()]).to(device)
        prediction = model.forward(input)
        # cm[i][np.argmax(prediction.cpu().detach().numpy())] += 1  # need to call detach or problems with gradients will occur
        predictions.append(np.argmax(prediction.cpu().detach().numpy())) # need to call detach or problems with gradients will occur

        feature = feature_extractor.forward(input)
        feature = feature.view(feature.size(0), -1)
        feature_vectors.append(feature.tolist()[0])
        targets.append(data.__getitem__(idx)[1])

# print accuracy, sensitivity, and specificity
# print(accuary_metrics(cm))

print(classification_report(predictions, targets))

# perform PCA for three dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(feature_vectors)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=targets, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()

# plot the confusion matrix
normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
