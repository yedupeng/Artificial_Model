import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from ResNet_Model import ResNet34

data_transform =transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img = Image.open('F:\Artificial Intelligence\\Network_Framework\data\\train\daisy\\5547758_eea9edfd54_n.jpg')
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)
try:
    json_file = open('./class_index.json','r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = ResNet34(num_class=5)
model_weight_path = "./ResNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    prdict = torch.softmax(output, dim=0)
    prdict_cla = torch.argmax(prdict).numpy()
print(class_indict[str(prdict_cla)], prdict[prdict_cla].item())
plt.show()