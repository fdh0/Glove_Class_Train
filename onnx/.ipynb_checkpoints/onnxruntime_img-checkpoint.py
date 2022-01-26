import onnxruntime as rt 
import numpy as  np
import cv2

sess = rt.InferenceSession("./1.onnx")

img = cv2.imread("1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

result = sess.run([label_name], {input_name:img.astype(np.float32)})[0]

print(result)

print(np.argmax(result))
