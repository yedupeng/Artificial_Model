import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from model import Vgg

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

model_name = "Vgg13"
model = Vgg(model_name=model_name, class_num=5, init_weights=False)
model_weight_path = "./VGG.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    prdict = torch.softmax(output, dim=0)
    prdict_cla = torch.argmax(prdict).numpy()
print(class_indict[str(prdict_cla)], prdict[prdict_cla].item())
plt.show()