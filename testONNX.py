import os

import torch
import torch.onnx
from torch.autograd import Variable
from model import efficientnetv2_s as create_model


def get_pytorch_onnx_model(original_model):
    onnx_model_path = "onnx"
    onnx_model_name = "efficientnetV2_917.onnx"
    os.makedirs(onnx_model_path, exist_ok=True)
    full_model_path = os.path.join(onnx_model_path, onnx_model_name)

    original_model.load_state_dict(torch.load("./weights/model-58.pth"))
    original_model.cuda()
    original_model.eval()
    generated_input = torch.randn(1, 3, 224, 224)
    generated_input=generated_input.cuda()

    torch.onnx.export(
        original_model,
        generated_input,
        full_model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
       )

    return full_model_path


def main():
    origin_model = create_model(num_classes=2)


    full_model_path = get_pytorch_onnx_model(origin_model)

    print("Pytorch  model was successfully converted:", full_model_path)


if __name__ == "__main__":
    main()