from shutil import copy, rmtree
import os
import random

def mdk(file_path):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.mkdir(file_path)

data_path = "F:\Artificial Intelligence\\Network_Framework\data\\"
origin_path = os.path.join(data_path+"flower_photos")

def main():
    random.seed(0)
    assert os.path.exists(origin_path), "path '{}' is not exist.".format(origin_path)
    flower_class = [cla for cla in os.listdir(origin_path)
                    if os.path.isdir(os.path.join(origin_path, cla))]
    train_root = os.path.join(data_path+"train")
    mdk(train_root)
    val_root = os.path.join(data_path+"val")
    mdk(val_root)
    for cla in flower_class:
        mdk(os.path.join(train_root, cla))
        mdk(os.path.join(val_root, cla))
    split_ratio = 0.1

    for cla in flower_class:
        cla_path = os.path.join(origin_path, cla)
        Images = os.listdir(cla_path)
        num = len(Images)
        evel_index = random.sample(Images, int(split_ratio*num))
        for index, image in enumerate(Images):
            if image in evel_index:
                Image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(Image_path, new_path)
            else:
                Image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(Image_path, new_path)
        print(cla+"is over!")

if __name__ == "__main__":
    main()