import os
from shutil import rmtree,copy
import random

def mdk(file_path):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.mkdir(file_path)

data_path = "F:\Artificial Intelligence\Git\\Network_Framework\data\\"
origin_path = os.path.join(data_path+"flower_photos")

def main():
    assert os.path.exists(origin_path), "path '{}' does not exists".format(origin_path)
    flower_classes = [cla for cla in os.listdir(origin_path)
                      if os.path.isdir(os.path.join(origin_path, cla))]
    train_root = os.path.join(data_path+"train")
    mdk(train_root)
    val_root = os.path.join(data_path+"val")
    mdk(val_root)
    for cla in flower_classes:
        mdk(os.path.join(train_root, cla))
        mdk(os.path.join(val_root, cla))
    split_ratio = 0.1

    for cla in flower_classes:
        cla_path = os.path.join(origin_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num*split_ratio))
        for index, image in enumerate(images):
            if image in eval_index:
                new_path = os.path.join(val_root, cla)
                image_path = os.path.join(cla_path, image)
                copy(image_path, new_path)
            else:
                new_path = os.path.join(train_root, cla)
                image_path = os.path.join(cla_path, image)
                copy(image_path, new_path)
        print(cla+"had finished!")

if __name__ == "__main__":
    main()