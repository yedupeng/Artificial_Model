import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms,datasets
from torch.utils.data import Dataset
from flower_model import Alexnet
import json
from torch.utils.tensorboard import SummaryWriter

#   调用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#   实例化tensorboard对象
rb_writer = SummaryWriter(log_dir="run/flower_experiment")
print(device)
#   图像预处理
data_transform = {
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    "val":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
#   路径选择
image_path = "E:\Git\Program_Artifical\data\\"
train_datasets = datasets.ImageFolder(root=image_path+"train",
                                  transform=data_transform["train"]
                                  )
val_datasets = datasets.ImageFolder(root=image_path+"val",
                                  transform=data_transform["val"]
                                  )
train_num = len(train_datasets)
val_num = len(val_datasets)
flower_list = train_datasets.class_to_idx                               # 得到对应标签所对应的索引
cla_dict = dict((val, key) for key, val in flower_list.items())           # 索引跟标签位置互换，利于字典查找
json_dir = json.dumps(cla_dict, indent=4)
with open("class_index.json", "w") as json_file:
    json_file.write(json_dir)
#  数据载入
batch_size = 32
train_loder = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
val_loder = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=0)
net = Alexnet(num_class=5,init_weight=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)
save_path = "./Alex.pth"
best_acc = 0.0
#  图像预处理  将模型写入tensorboard
init_image = torch.zeros((1,3,224,224),device = device)
rb_writer.add_graph(net,init_image)
for epoch in range(10):
    net.train()
    runnning_loss = 0.0
    for step, data in enumerate(train_loder, start=0):
        text_image, label = data
        optimizer.zero_grad()
        output = net(text_image.to(device))
        loss = loss_function(output, label.to(device))
        loss.backward()
        optimizer.step()

        runnning_loss += loss.item()
        rate = (step+1)/len(train_loder)
        a = "*"*int(rate*50)
        b = "."*int((1-rate)*50)
        print("\rtrain loss:{:3.0f}%[{}->{}]{:3f}".format(int(rate*100), a, b, loss), end="")

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loder:
            val_image, val_label = val_data
            acc_ouput = net(val_image.to(device))
            predict_y = torch.max(acc_ouput, dim=1)[1]
            acc += (predict_y == val_label.to(device)).sum().item()
        acc_text = acc/val_num
        if acc_text>best_acc:
            best_acc = acc_text
            torch.save(net.state_dict(),save_path)
        print("eproch %d train_loss: %.3f  test_acc:%.3f"%
              (epoch+1,runnning_loss/step, acc/val_num))
    print("over!")
    tags = ["train_loss","accuracy","learning_rate"]
    rb_writer.add_scalar(tags[0],loss,epoch)
    rb_writer.add_scalar(tags[1],acc/val_num,epoch)
    rb_writer.add_scalar(tags[2],optimizer.param_groups[0]["lr"],epoch)