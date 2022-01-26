import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"
    #print(img_size[num_model][1])
    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    filePath = '/home/feng/deeplearning/deeplearning/Projects/Classify/1_images_folder/images_gloves_picked_202009/train/left/'
    name = os.listdir(filePath)
    count =0
    ERRcount = 0

    true_label = 'left'

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights/model-8.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    for i in name:
        count +=1
        # load image
        img_path = filePath+i
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)

        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './predict/class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

       
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            
            print(output)
            
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        pre = class_indict[str(predict_cla)]

        if (pre != true_label):
            # print('***********************************')
            print('pre:'+pre,'true:'+true_label,img_path)
            ERRcount += 1

        print('pre:'+pre,'true:'+true_label,str(count)+'/'+str(len(name)))

    print('-----------------------------------ERRcount----------------------------------------')
    print(ERRcount)
if __name__ == '__main__':
    main()
