import cv2 as cv

# cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
#cvNet = cv.dnn.readNetFromTensorflow('k_t.pb')
cvNet = cv.dnn.readNetFromONNX('1.onnx')

img = cv.imread('1.jpg',0)
rows = img.shape[0]
cols = img.shape[1]

cvNet.setInput(cv.dnn.blobFromImage(img, 1/255.0, size=(224, 224), swapRB=False, crop=False))

cvOut = cvNet.forward()

print(cvOut)

cv.imshow('img', img)
