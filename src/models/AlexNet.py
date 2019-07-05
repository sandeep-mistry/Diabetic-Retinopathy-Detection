import torch
import torchbearer
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader

from RetDataset import RetDataset

num_epochs = 10
batch_size = 32
train_path = "data/train_512_balanced/"
test_path = "data/test_512/"
train_labels_path = "data/trainLabels_balanced.csv"
test_labels_path = "data/testLabels.csv"
save_weights_path = "weights/je_alexnet_chopped_0905"
cuda = 2
lr = 0.0001
l2 = 0.0005

transform = transforms.Compose([
    transforms.ToTensor()
])

print('Running AlexNet unfrozen training for', num_epochs, 'epochs')
print('Loading data')
train_set = RetDataset(train_labels_path, train_path, transform=transform)
test_set = RetDataset(test_labels_path, test_path, transform=transform)
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = "cuda:" + str(cuda)
    print("Running with GPU Acceleration")
else:
    print("Running on CPU")
    device = "cpu"

# Setup model, replacing final layer with 5 outputs instead of 1000
print('Setting up model')
model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 5)
model.to(device)

# Setup training
print('Setting up training')
loss_function = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=l2, amsgrad=False)
trial = torchbearer.Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)
trial.with_generators(trainloader, test_generator=testloader)

# Train
print('Running training')
trial.run(epochs=num_epochs)

# Evaluate
print("Saving weights")
torch.save(model.state_dict(), save_weights_path)
print('Evaluating')
model.eval()
results = trial.evaluate(data_key=torchbearer.TEST_DATA)
print(results)

