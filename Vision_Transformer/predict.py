from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
import json
import torch
from model import vit_base_patch16_224_in21k

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

Image = Image.open("F:\Artificial Intelligence\\Network_Framework\data\\train\daisy\\5547758_eea9edfd54_n.jpg")
plt.imshow(Image)
Image = data_transform(Image)
Image = torch.unsqueeze(Image, dim=0)

try:
    json_file = open("F:\Artificial Intelligence\\Network_Framework\Vit_Transformer\VitTransformer.json")
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = vit_base_patch16_224_in21k(num_class=5)
model_path = "F:\Artificial Intelligence\\Network_Framework\Vit_Transformer\VitTransformer.pth"
mis_key, unexpected_key = model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(Image))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    print(class_dict[str(predict_cla)], predict[predict_cla].item())
    plt.show()