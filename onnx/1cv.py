import cv2
# opencv 推理
net = cv2.dnn.readNetFromONNX("1.onnx")  # 加载训练好的识别模型
image = cv2.imread("1.jpg")  # 读取图片
blob = cv2.dnn.blobFromImage(image)  # 由图片加载数据 这里还可以进行缩放、归一化等预处理
net.setInput(blob)  # 设置模型输入
out = net.forward()  # 推理出结果

