import os
from shutil import rmtree, copy
import random

#  创建文件夹，若原文件存在则删除文件重新创建
def mk_file(file_path:str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.mkdir(file_path)

def main():
    random.seed(0)                                                                           #  保证随机数可复现
    split_ratio = 0.1                                                                        #  测试集数据的比例
    data_root = "F:\Artificial Intelligence\Git\\Network_Framework\data\\"
    origin_flower = os.path.join(data_root+"flower_photos")
    assert os.path.exists(origin_flower), "path '{}' does not exist.".format(origin_flower)
    flower_class = [cla for cla in os.listdir(origin_flower)                                 #  判断是否为目录并将目录名字传给cla
                    if os.path.isdir(os.path.join(origin_flower,cla))]

    train_root = os.path.join(data_root+"train")                                             #  训练集文件
    mk_file(train_root)
    for cla in flower_class:
        mk_file(os.path.join(train_root,cla))                                                #  训练集中创建类别文件

    val_root = os.path.join(data_root+"val")                                                 #  测试集文件
    mk_file(val_root)
    for cla in flower_class:                                                                 #  测试集中创建类别文件
        mk_file(os.path.join(val_root,cla))

    for cla in flower_class:                                                                 #  图片搬运
        cla_path = os.path.join(origin_flower,cla)
        images = os.listdir(cla_path)
        num = len(images)                                                                    #  提取样本个数
        print(num)
        eval_index = random.sample(images, k=int(num*split_ratio))
        for index, image in enumerate(images):
            if image in eval_index:                                                          #  根据比例转移图片至train、val
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path,new_path)

        print("process done!")

if __name__ == '__main__':
    main()