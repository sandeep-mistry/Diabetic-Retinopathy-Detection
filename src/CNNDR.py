import argparse

import torch
import torchbearer
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim, nn
from torch.utils.data import DataLoader

from RetCNN import RetCNN, RetResNet
from RetDataset import RetDataset
from torchbearer.callbacks import EarlyStopping
import torch.cuda as cutorch

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

load_weights = False
save_weights = False

if (args.load_weights_name != None):
    load_weights = True
if (args.save_weights_name != None):
    save_weights = True

# use the dataset to get train and test batches.

if args.transform == 1 :
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),  # convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

elif args.transform == 2:
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

train_set = RetDataset(args.train_csv, args.train_images, transform=transform)
test_set = RetDataset(args.test_csv, args.test_images, transform=transform)

trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

if torch.cuda.is_available():
    device = "cuda:" + str(args.cuda)
    print("Running with GPU Acceleration")

    # print(cutorch.device_count(), 'CUDA devices found')
    # for i in range(cutorch.device_count()):
    #     print(i, cutorch.getMemoryUsage(i))

else:
    print("Running on CPU")
    device = "cpu"

if args.alexnet:
    print("Using Alexnet")
    model = models.alexnet(pretrained=args.pretrained) if args.pretrained else models.alexnet()
    set_parameter_requires_grad(model, args.freezepretrained)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 5)

elif args.inception:
    print("Using Inception")
    model = models.inception_v3(pretrained=args.pretrained) if args.pretrained else models.alexnet()

elif args.vgg:
    print("Using VGG 11 with BN")
    model = models.vgg11_bn(pretrained=args.pretrained)
    set_parameter_requires_grad(model, args.freezepretrained)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 5)

elif args.resnet:
    print("Using Resnet 18")
    model = models.resnet18(pretrained=True)
    print(args.freezepretrained)
    set_parameter_requires_grad(model, args.freezepretrained)
    model.fc = nn.Linear(512, 5)

elif args.retresnet:
    print("Using Resnet 18 with mlp on the end")
    model = RetResNet()

else:
    print("Using RetCNN")
    model = RetCNN()

# define the loss function and the optimiser, and weight the classes for the imbalance in our dataset such that all
 # class are equally balanced when taking account loss in mislabelling.
if not args.balanced:
    print("Loss function will compensate for imbalanced dataset!")
    label_ratios, label_counts = train_set.get_ratios()
    label_weights = 1 / label_ratios
    label_weights = torch.from_numpy(label_weights).float().to(device)

    loss_function = nn.CrossEntropyLoss(weight=label_weights)
else:
    loss_function = nn.CrossEntropyLoss()

model.to(device)

if load_weights:
    print("Loading weights")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.load_weights_name))
    else:
        model.load_state_dict(torch.load(args.load_weights_name, map_location='cpu'))

optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2, amsgrad=args.ams_grad)

# Construct a trial object with the model, optimiser and loss.
# Also specify metrics we wish to compute.
trial = torchbearer.Trial(model, optimiser, loss_function, metrics=['loss', 'accuracy']).to(device)

# Provide the data to the trial
trial.with_generators(trainloader, test_generator=testloader)

# Train
if args.train:
    trial.run(epochs=args.epochs)
else:
    model.eval()

# test the performance
results = trial.evaluate(data_key=torchbearer.TEST_DATA)
print(results)
print("Saving weights")
if save_weights: torch.save(model.state_dict(), args.save_weights_name)
