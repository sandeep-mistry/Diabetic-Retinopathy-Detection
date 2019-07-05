import os

import torch
import torchvision.models as models
from torch import nn
from tqdm import tqdm

from src.evaluation.Evaluate import Evaluator, SubmissionEvaluator


class CNNEvaluator(Evaluator):

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def get_predictions(self):
        self.model.to(self.device)
        all_predictions = []
        truth_list = []
        for i, batch in tqdm(enumerate(self.test_loader),
                             total=self.num_points // self.batch_size, desc='Predicting'):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predictions = torch.max(outputs.data, 1)
            all_predictions.extend(predictions.cpu())
            truth_list.extend(labels.cpu())
        return all_predictions


if __name__ == '__main__':
    cuda = 2
    weights_filename = "je_alexnet_chopped_0905"
    weights_path = os.path.join('weights', weights_filename)

    submission_filename = weights_filename + "_submission.csv"
    submission_path = os.path.join('submissions', submission_filename)

    print('Running model evaluation script')

    if os.path.exists(submission_path):
        print('Submission file found, evaluating...')
        evaluator = SubmissionEvaluator(submission_path)
        evaluator.evaluate()
    else:
        print('Submission file not found, using model evaluation...')
        print('Setting up model')
        model = models.alexnet(pretrained=True)
        model.eval()
        model.classifier[6] = nn.Linear(4096, 5)

        print("Loading weights")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))

        if torch.cuda.is_available():
            print("Running with GPU Acceleration")
            device = "cuda:" + str(cuda)
        else:
            print("Running on CPU")
            device = "cpu"

        print("Evaluating model")
        eval = CNNEvaluator(model, device)
        predicted_labels, accuracy, _, qwk = eval.evaluate()

        print('Saving submission')
        eval.save_submission(predicted_labels, submission_path)