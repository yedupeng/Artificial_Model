import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from flower_model import Alexnet

data_transform =transforms.Compose([                                         #图像预处理
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
img = Image.open('F:\Artificial Intelligence\Git\\Network_Framework\data\\flower_photos\\roses\\22679076_bdb4c24401_m.jpg')
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)
try:
    json_file = open('./class_index.json','r')                              #加载标签文件
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = Alexnet(num_class=5)                                                #传入网络及权重文件
model_weight_path = "./Alex.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))                                      #图片传进网络并压缩维度
    prdict = torch.softmax(output, dim=0)                                   #预测准确度和标签
    prdict_cla = torch.argmax(prdict).numpy()
print(class_indict[str(prdict_cla)], prdict[prdict_cla].item())
plt.show()