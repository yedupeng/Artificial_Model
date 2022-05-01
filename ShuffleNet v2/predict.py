import json
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import Shuffle_NetV2_x1_0

data_transform = transforms.Compose(
    [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

img = Image.open('F:\Artificial Intelligence\\Network_Framework\data\\train\daisy\\5547758_eea9edfd54_n.jpg')
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open("./ShuffleNet.json", "r")
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = Shuffle_NetV2_x1_0(num_classes=5)
model_weight = "F:\Artificial Intelligence\\Network_Framework\ShuffleNet v2\ShuffleNet.pth"
model.load_state_dict(torch.load(model_weight))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    print(class_indict[str(predict_cla)], predict[predict_cla].item())
    plt.show()


