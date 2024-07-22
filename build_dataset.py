import cv2
from sklearn.model_selection import train_test_split
import os


def make_set(datafold_name, source_dataset):
    img_list = os.listdir(source_dataset)
    if not os.path.exists(datafold_name):
        os.makedirs(datafold_name)
    traindir = os.path.join(datafold_name, "train")
    testdir = os.path.join(datafold_name, "test")
    if not os.path.exists(traindir):
        os.makedirs(traindir)
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    train, test = train_test_split(img_list, test_size=0.2, random_state=42)
    for item in train:
        srcimgpath = os.path.join(source_dataset, item)
        img_preprocess(srcimgpath, write_path=traindir)
    for item in test:
        srcimgpath = os.path.join(source_dataset, item)
        img_preprocess(srcimgpath, write_path=testdir)


def img_preprocess(img_filepath, write_path):
    _, img_name = os.path.split(img_filepath)
    src_img = cv2.imread(img_filepath)
    resized_img = cv2.resize(src_img, dsize=[256, 256])
    cv2.imwrite(os.path.join(write_path, img_name), resized_img)


if __name__ == '__main__':
    foldname = "data"
    source_dataset = "Kodak24"
    make_set(foldname, source_dataset)
