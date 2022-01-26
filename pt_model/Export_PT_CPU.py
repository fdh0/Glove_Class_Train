import torch
import torchvision


from model import efficientnetv2_s as create_model
# 初始化神经网络
device = torch.device('cpu')
#net.to(device)

net = create_model(num_classes=2)
net.load_state_dict(torch.load("../weights/model-9.pth",map_location='cpu'))
#net.cuda()
#net.cpu()
net.to(device)
net.eval()


x = torch.randn(1, 3, 224, 224)
#x = x.cuda()

with torch.no_grad(): #不要梯度，不然显存会爆炸
    # 导出模型
    m = torch.jit.trace(net, x)
    m.save("efficientnetv2_917_cpu3.pt")
