import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from MobileNet_model import MobileNet_V2

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img = Image.open("F:\Artificial Intelligence\\Network_Framework\data\\train\daisy\\5547758_eea9edfd54_n.jpg")
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

try:
    json_file = open('F:\Artificial Intelligence\\Network_Framework\RexNet\class_index.json', 'r')
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = MobileNet_V2(num_class=5)
model_weight_path = "F:\Artificial Intelligence\\Network_Framework\Mobile_Net\MobileNet.pth"
model.load_state_dict(torch.load(model_weight_path))
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_dict[str(predict_cla)], predict[predict_cla].item())
plt.show()
