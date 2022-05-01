import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from Google_model import GoogleNet

data_transform =transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
img = Image.open('F:\Artificial Intelligence\Git\\Network_Framework\data\\flower_photos\\roses\\22679076_bdb4c24401_m.jpg')
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)
try:
    json_file = open('./class_index.json','r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = GoogleNet(num_classes=5, aux_enable=False)
model_weight_path = "./Google.pth"
miss_key, unexpected_key = model.load_state_dict(torch.load(model_weight_path), strict=False)
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    prdict = torch.softmax(output, dim=0)
    prdict_cla = torch.argmax(prdict).numpy()
print(class_indict[str(prdict_cla)])
plt.show()