import os
from shutil import copy, rmtree
import random

def mdk(file_path):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.mkdir(file_path)

data_root = "F:\Artificial Intelligence\\Network_Framework\data\\"
flower_path = os.path.join(data_root+"flower_photos")

def main():
    assert os.path.exists(flower_path), "path {} is ont exist".format(flower_path)
    flower_class = [cla for cla in os.listdir(flower_path)
                    if os.path.isdir(os.path.join(flower_path, cla))]
    train_root = os.path.join(data_root+"train")
    val_root = os.path.join(data_root+"val")
    mdk(train_root)
    mdk(val_root)
    for cla in flower_class:
        mdk(os.path.join(train_root, cla))
        mdk(os.path.join(val_root, cla))

    for cla in flower_class:
        data_path = os.path.join(flower_path,cla)
        Images = os.listdir(data_path)
        ratio = 0.1
        num = len(Images)
        val_index = random.sample(Images, int(num*ratio))
        for index, Image in enumerate(Images):
            if Image in val_index:
                Image_path = os.path.join(data_path, Image)
                new_path = os.path.join(val_root, cla)
                copy(Image_path, new_path)
            else:
                Image_path = os.path.join(data_path, Image)
                new_path = os.path.join(train_root, cla)
                copy(Image_path, new_path)
        print(cla + "is over!")
if __name__ == "__main__":
    main()