import glob
import cv2
from tqdm import tqdm
import csv
import numpy as np
from collections import Counter
import os

train_dir = 'data/train_512/'
raw_train_path = train_dir + '*.jpeg'
train_labels_path = 'data/trainLabels.csv'
target_dir = 'data/train_512_balanced_400/'
target_labels_path = 'data/trainLabels_balanced_400.csv'
target_class_size = 200
classes = range(5)


def load_data(labels_path, num_images):
    file_distribution = {}
    image_distribution = {}
    with open(labels_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for name, clz in tqdm(reader, total=num_images):
            clz = int(clz)
            if clz not in classes:
                continue
            if clz not in file_distribution:
                file_distribution[clz] = []
                image_distribution[clz] = []
            file_path = train_dir + name + '.jpeg'
            file_distribution[clz].append(name)
            image_distribution[clz].append(cv2.imread(file_path))
    return file_distribution, image_distribution


def save_samples(directory, files, images, clz):
    if not os.path.exists(directory):
        os.mkdir(directory)
    for i in tqdm(range(len(files)), desc='Class ' + str(clz)):
        cv2.imwrite(files[i], images[i])


def save_labels(file_distribution):
    with open(target_labels_path, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(('image', 'level'))
        for i in classes:
            files = file_distribution[i]
            for file in files:
                file = file[file.rindex('/') + 1: -5]
                writer.writerow((file, i))


def sample(files, images, sample_size, starting_index):
    sampled_files = []
    sampled_images = []
    diff = sample_size - len(files)
    if diff > 0:  # Oversampling required
        sampled_files = [target_dir + f + '.jpeg' for f in files]  # Keep all current files
        sampled_images = images[:]  # Keep all current images
        selection_indices = np.random.choice(range(len(files)), size=diff)
        for index in selection_indices:
            o_name = files[index]
            sample_name = target_dir + str(starting_index) + o_name[o_name.rindex('_'):] + '.jpeg'
            starting_index += 1
            sampled_files.append(sample_name)
            sampled_images.append(images[index])
    elif diff < 0:  # Reduced sampling required
        selection_indices = np.random.choice(range(len(files)), size=sample_size, replace=False)
        for index in selection_indices:
            o_name = files[index]
            sample_name = target_dir + o_name + '.jpeg'
            sampled_files.append(sample_name)
            sampled_images.append(images[index])
    else:
        sampled_files = files[:]
        sampled_images = images[:]

    return sampled_files, sampled_images, starting_index


if __name__ == '__main__':
    _num_images = len(glob.glob(raw_train_path))
    print('Found', _num_images, 'images, loading...')
    _starting_index = np.max([int(s[s.index('_') + 5:s.rindex('_')]) for s in glob.glob(raw_train_path)]) + 1
    _file_distribution, _image_distribution = load_data(train_labels_path, _num_images)
    print('Class distribution:')
    for i in classes:
        print(i, len(_file_distribution[i]))

    print('Sampling')
    for i in classes:
        sampled_files, sampled_images, _starting_index = sample(_file_distribution[i], _image_distribution[i], target_class_size, _starting_index)
        _file_distribution[i] = sampled_files
        _image_distribution[i] = sampled_images

    print('New distribution:')
    for i in classes:
        print(i, len(_file_distribution[i]))

    # Check for bad class sizes, duplicates, and bad file names
    for i in classes:
        files = _file_distribution[i]
        images = _image_distribution[i]
        assert len(files) == len(set(files)) == target_class_size
        assert len(images) == target_class_size
        for file in files:
            assert file.startswith(target_dir)
            assert file.endswith('right.jpeg') or file.endswith('left.jpeg')
    print('File name checks okay')

    # Save images
    print('Saving training images')
    for i in classes:
        save_samples(target_dir, _file_distribution[i], _image_distribution[i], i)

    # Save labels
    print('Saving training labels')
    save_labels(_file_distribution)

    print('All done')
