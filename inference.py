import numpy as np
import os
import tensorflow as tf
import cv2
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video ")


args = vars(ap.parse_args())
video_path = args["video"]

import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='model_1010_2.tflite')
interpreter.allocate_tensors()

interpreter2 = tflite.Interpreter(model_path='model_cmnist_finetune_1011_128x64.tflite')
interpreter2.allocate_tensors()

def preprocess_image(image_path, input_size):
    """Preprocess the input image to feed to the TFLite model"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    original_image = img
    resized_img = tf.image.resize(img, input_size)
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image

def set_input_tensor(interpreter, image):
    """Set the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image
    
def get_output_tensor(interpreter, index):
    """Retur the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def subblock_detection(interpreter, image, threshold,
                       input_height, input_width,
                       target_w = 640, target_h = 480,
                       target_blocksize = 64
                      ):
    
    img = tf.convert_to_tensor(image) 
  
    img = tf.cast(img, tf.uint8)
     
    resized_img = tf.image.resize(img, (input_height, input_width))
     
    preprocessed_image = resized_img[tf.newaxis, :]
     
    
    # Load the input image and preprocess it     

    set_input_tensor(interpreter, preprocessed_image)
    interpreter.invoke()

    # Get all outputs from the model
    out = get_output_tensor(interpreter, 0)

    
    subblock_mask = out > threshold
    
    block_size = int(target_blocksize* image.shape[1]/target_h)
     
    s1 = np.sum(subblock_mask, axis = 0)
    s2 = np.sum(subblock_mask, axis = 1)

    xRange = np.where(s1 > 0)[0]
    yRange = np.where(s2 > 0)[0]
    
    #print(s1)
    #print(s2)

    if len(xRange) > 0 and len(yRange) > 0:
        x0 = xRange[0]*block_size
        x1 = (xRange[-1]+1)*block_size
 
        y0 = yRange[0]*block_size
        y1 = (yRange[-1]+1)*block_size
        
        return (x0, y0, x1, y1)
    
    return []
    

def classification(image, interpreter2, targetsize = (128,64)):
    _, input_height, input_width, _ = interpreter2.get_input_details()[0]['shape'] 
    img = tf.convert_to_tensor(image) 
     
    img = tf.cast(img, tf.uint8)
     
    resized_img = tf.image.resize(img, (input_height, input_width))
     
    preprocessed_image = resized_img[tf.newaxis, :]
    
    set_input_tensor(interpreter2, preprocessed_image)
    interpreter2.invoke()

    # Get all outputs from the model
    out2 = get_output_tensor(interpreter2, 0)
    
    return out2



threshold = 0.25
classifier_target_h = 128
classifier_target_w = 64
# read video file
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

showText = ['', 'MISALIGNED']
classifier_target_h = 128
classifier_target_w = 64
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

while cap.isOpened() and ret :
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_copy = image.copy()
    image = image.reshape(1,image.shape[0],image.shape[1],1)
    
    bbox = subblock_detection(interpreter, image, threshold,
                       input_height, input_width,
                       target_w = 640, target_h = 480,
                       target_blocksize = 64)
    
  
    
    if len(bbox) > 0:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        image2 = image_copy[bbox[1]:bbox[3],bbox[0]:bbox[2]]
         
        image2 = cv2.resize(image2, (classifier_target_h,classifier_target_w), interpolation=cv2.INTER_LINEAR)
        image2=image2.reshape(1,classifier_target_h,classifier_target_w,1)
        out2 = classification(image2, interpreter2, targetsize = (classifier_target_h,classifier_target_w))
        out2Idx = int(out2 > 0.5)
         
        cv2.putText(frame, showText[out2Idx], (10, 40), cv2.FONT_HERSHEY_COMPLEX,
                              1, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    ret, frame = cap.read() 
    if cv2.waitKey(10) & 0xFF == ord('q'):
     
        break
   
    
cap.release()
cv2.destroyAllWindows()


