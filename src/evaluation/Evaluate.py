import csv
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from RetDataset import RetDataset

test_image_dir = 'data/test_512/'
test_labels_path = 'data/testLabels.csv'

transform = transforms.Compose([
    transforms.ToTensor()
])


class Evaluator(ABC):

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.test_set = RetDataset(test_labels_path, test_image_dir, transform=transform)
        self.test_loader = DataLoader(self.test_set, batch_size, shuffle=False)
        self.num_points = len(self.test_set)

    @abstractmethod
    def get_predictions(self):
        pass

    def evaluate(self):
        predicted_labels = self.get_predictions()
        correct_labels = self.test_set.get_labels().tolist()
        accuracy = accuracy_score(correct_labels, predicted_labels)
        f1score = f1_score(correct_labels, predicted_labels, average='weighted')
        qwk = cohen_kappa_score(correct_labels, predicted_labels, weights='quadratic')
        cm = np.zeros((5, 5))
        true_dist = np.zeros(5)

        for i in range(len(predicted_labels)):
            cm[predicted_labels[i]][correct_labels[i]] += 1
            true_dist[predicted_labels[i]] += 1

        for i in range(5):
            for j in range(5):
                cm[i][j] /= true_dist[i]

        cm = confusion_matrix(correct_labels, predicted_labels)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axis = plt.subplots(1, 1)
        plot = axis.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        axis.set_title("Confusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.xaxis.set_ticks_position("top")
        axis.xaxis.set_label_position("top")
        plt.colorbar(plot)
        plt.show()

        print('Accuracy:', accuracy)
        print('F1 Score:', f1score)
        print('QuadWeightedKappa:', qwk)
        print(cm)
        return predicted_labels, accuracy, f1score, qwk

    def save_submission(self, predictions: List[int], submission_path):
        image_names = self.test_set.get_img_names()
        print('Writing submission to', submission_path)

        if not os.path.exists('submissions'):
            os.mkdir('submissions')

        with open(submission_path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['image', 'level'])
            for i in tqdm(range(len(predictions))):
                writer.writerow([image_names[i], predictions[i].detach().numpy()])


class SubmissionEvaluator(Evaluator):

    def __init__(self, submission_path):
        super().__init__()
        self.submission_path = submission_path

    def get_predictions(self):
        predictions = []
        with open(self.submission_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in tqdm(reader, total=self.num_points, desc='Parsing submission file'):
                predictions.append(int(row[1]))
        return np.array(predictions)
