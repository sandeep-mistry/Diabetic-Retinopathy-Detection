import cv2
import glob
import numpy as np
import os

read_images = 'train/*jpeg'
write_folder = 'processed'
PRESENT = False


def make_dir(directory):
    try:
        os.mkdir(directory)
    except OSError:
        print("Creation of the directory %s failed" % directory)
    else:
        print("Successfully created the directory %s " % directory)


# save into path directory a matrix of images, each row being an image
def save_samples(directory, images):
    for idx in range(len(images)):
        cv2.imwrite(str(directory) + '/sample' + str(idx) + '.jpeg', images[idx])


# augment images in a directory with rotation by degrees amount
def augment_rotation(folder, degrees):
    # read the images
    images = [cv2.imread(file) for file in glob.glob(folder)]
    rotated_images = []

    for image in images:
        num_rows, num_cols = image.shape[:2]

        # rotate the images
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), degrees, 1)
        img_rotation = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
        rotated_images.append(img_rotation)

        if PRESENT:
            cv2.imshow('Rotation', img_rotation)
            cv2.waitKey(0)

    # write the images, same folder?
    make_dir("augment")
    save_samples("augment", rotated_images)
    return 0


def image_loading(folder):
    count = 0
    resized_image = []
    new_images = []
    image_norm = []
    image_crop = []
    image_sizing = []
    images = [cv2.imread(file) for file in glob.glob(folder)]
    greyimg = []
    threshold = 25

    # converting to greyscale for cropping purposes
    for j in range(len(images)):
        greyimg.append(cv2.cvtColor(images[j], cv2.COLOR_BGR2GRAY))

    # cropping the image to get rid of extra black border
    for j in range(len(images)):  # cropping image to remove extra black borders
        hStart = 0
        hEnd = greyimg[j].shape[0]
        vStart = 0
        vEnd = greyimg[j].shape[1]

        # get row and column maxes for each row and column
        hMax = greyimg[j].max(1)
        vMax = greyimg[j].max(0)

        hDone_flag = False
        vDone_flag = False

        # go through the list of max and begin where the pixel value is greater
        # than the threshold
        for i in range(hMax.size):
            if not hDone_flag:
                if hMax[i] > threshold:
                    hStart = i
                    hDone_flag = True

            if hDone_flag:
                if hMax[i] < threshold:
                    hEnd = i
                    break

        for i in range(vMax.size):
            if not vDone_flag:
                if vMax[i] > threshold:
                    vStart = i
                    vDone_flag = True

            if vDone_flag:
                if vMax[i] < threshold:
                    vEnd = i
                    break
        image_crop.append(images[j][hStart:hEnd, vStart:vEnd])
        image_sizing.append(cv2.resize(image_crop[j], (512, 512)))

    # adding left and right images together
    for j in range(len(image_sizing)):
        if (j % 2) == 0:
            new_images.append(np.concatenate((image_sizing[j], image_sizing[j + 1]), axis=1))

    # resizing images to 128,128, colour normalisation and image de-noising
    for i in range(len(new_images)):
        # height, width = new_images[i].shape[:2]
        count += 1
        resized_image.append(cv2.resize(new_images[i], (512, 512)))
        image_norm.append(
            cv2.fastNlMeansDenoisingColored(cv2.normalize(resized_image[i], None, 0, 255, cv2.NORM_MINMAX)))
        # height0, width0 = resized_image.shape[:2]
    print('Total Training Images = ', count)
    return image_norm


augment_rotation(read_images, 30)
images = image_loading(read_images)

if PRESENT:
    for i in range(len(images)):
        cv2.imshow("Show by CV2", images[i])
        cv2.waitKey(0)

save_samples(write_folder, images)
