import torch
import torchvision.transforms as transforms
from RetDataset import RetDataset
from PIL import Image

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

csv_file = 'data/testLabels.csv'
image_root = 'data/test_512'

data = RetDataset(csv_file, image_root, transform=transform)
input = torch.Tensor([data.__getitem__(9)[0].tolist()])
toimage = transforms.ToPILImage()
output = toimage(input[0])
print(output)
output.show()
