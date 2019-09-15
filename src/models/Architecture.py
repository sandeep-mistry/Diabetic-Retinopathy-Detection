"""

This file is intended to replicate the training of 'Convolutional Neural Networks for Diabetic Retinopathy' Frans Coenen et. al
This includes using the network specified as well as the dynamic learning rates throughout training.

"""
import torch
import torchbearer
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader
import torchvision.models as models

from RetDataset import RetDataset

train_csv = 'train.csv'
test_csv = 'test.csv'
train_images = 'train_sub/'
test_images = 'test_sub/'

weight_decay = 0.0005       # lambda for L2 reg
epochs = 10                # this currently, is not the same number of epochs found in the paper this implementation is a draft version.
batch_size = 16             # small enough to not be killed by OS for taking too much memory, experiment with this value.

# use the dataset to get train and test batches.
transform = transforms.Compose([
    transforms.ToTensor()  # convert to tensor
])

train_set = RetDataset(train_csv, train_images, transform=transform)
test_set = RetDataset(test_csv, test_images, transform=transform)
trainloader = DataLoader(train_set, batch_size=16, shuffle=True)
testloader = DataLoader(test_set, batch_size=16, shuffle=True)

model = models.vgg16(pretrained=True)
device = "cuda:0"
if torch.cuda.is_available():
    model.cuda()
    print("Running with GPU Acceleration")
else:
    print("Running on CPU")

# define the loss function and the optimiser
loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), weight_decay=weight_decay)

# Construct a trial object with the model, optimiser and loss.
# Also specify metrics we wish to compute.
trial = torchbearer.Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
# Provide the data to the trial
trial.with_generators(trainloader, test_generator=testloader)

# Run 10 epochs of training
trial.run(epochs=epochs)

# test the performance
results = trial.evaluate(data_key=torchbearer.TEST_DATA)
print(results)
