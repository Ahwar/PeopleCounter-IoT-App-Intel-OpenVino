#!/usr/bin/env python3
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    """
    Connect to the MQTT client
    
    Returns:
        client: Connected client
    """
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client("41")
    client.connect(MQTT_HOST,port=MQTT_PORT, keepalive=MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocessing(image, w, h):
    """
    Preprocess input image for inference.

    Changes image shape from [HxWxC] to [1xCxHxW].
    You may need to change code according to your model requirements.
    Parameters:
        image (numpy.ndarray): Image you want to preprocess
        w (int): Width of Image after PreProcessing.
        h (int): Height of Image after PreProcessing.
    
    Returns:
        image (numpy.ndarray): PreProcessed Image
        
    """
    image = cv2.resize(image, (w, h))
    image = np.transpose(image, (2, 0, 1))
    image = image.reshape(1, 3, h, w)
    
    return image

def apply_threshold(outputs, frame, threshold):
    """
    Find number of People from model Output and Draw Bounding boxes
    
    Filter Detected people with confidence more then Threshold.
    Draw Bounding boxes around all detected People
    Calculate number of people in frame
    
    Parameters:
        output (numpy.ndarray): Model output.
        frame (numpy.ndarray): Original Frame captured from opencv input stream.
        threshold (float): The minimum threshold for detections.
    
    Returns:
        current_count (int): Number of People Detected.
        image (numpy.ndarray): Actual Frame with detected bounding boxes
    """
    image = frame
    frame_height, frame_width = frame.shape[:-1]
    current_count = 0
    for output in outputs:
        if output[2] > threshold:
            ### draw bounding box around person
            start_point = ( int(output[3] * frame_width), int(output[4] * frame_height) )
            end_point = ( int(output[5] * frame_width), int(output[6] * frame_height) )
            image = cv2.rectangle(frame, start_point, end_point, (13,255,0), 2)
            # count the number of detected people
            current_count += 1
    return image, current_count
        
        

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    
    . Load Model
    . Capture input stream from either camera, video or Image
    . Run Async inference per frame.
    . Calculate Stats and send image and stats to MQTT or FFMPEG server
    
    Parameters:
        args: Command line arguments parsed by `build_argparser()`.
        client: connected MQTT client
        threshold (float): The minimum threshold for detections.
    
    Returns:
        None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.exec_network = infer_network.load_model\
        (args.model, args.device, args.cpu_extension)
    # extract information about model input layer
    (b, c, input_height, input_width) = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # extenstion of input file
    input_extension = os.path.splitext(args.input)[1].lower()
    supported_vid_exts = ['.mp4', '.mpeg', '.avi', '.mkv']
    supported_img_exts = [".bmp",".dib", ".jpeg", ".jp2", ".jpg", ".jpe",\
        ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"]
    single_image_mode = False
    # if input is camera
    if args.input.upper() == 'CAM':
        capture = cv2.VideoCapture(0)
    
    # if input is video
    elif input_extension in supported_vid_exts:
        capture = cv2.VideoCapture(args.input)
    
    # if input is image
    elif input_extension in supported_img_exts:
        single_image_mode = True
        capture = cv2.VideoCapture(args.input) 
        capture.open(args.input)
    else:
        sys.exit("FATAL ERROR : The format of your input file is not supported" \
                     "\nsupported extensions are : " + ", ".join(supported_exts))
    prev_count = 0
    total_persons = 0
    ### TODO: Loop until stream is over ###
    while (capture.isOpened()):
        ### TODO: Read from the video capture ###
        ret, frame = capture.read()
        if not ret:
            break
        ### TODO: Pre-process the image as needed ###
        image = preprocessing(frame, input_width, input_height)
        ### TODO: Start asynchronous inference for specified request ###
        start_time = time.time()
        # run inference
        infer_network.exec_net(image)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            infer_time = time.time() - start_time
            ### TODO: Get the results of the inference request ###
            outputs = infer_network.get_output()[0][0]
            ### Take model output and extract number of detections with confidence exceeding threshold
            ### and draw bounding boxes around detections
            out_image, current_count = apply_threshold(outputs, frame, prob_threshold)
            
            # show inference time on image
            cv2.putText(out_image, "inference time: {:.5f} ms".format(infer_time), (30, 30),\
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            
            ### TODO: Extract any desired stats from the results ###
            # when any person exit
            if current_count < prev_count:
                ### Topic "person/duration": key of "duration" ###
                # send duration to mqtt server client
                client.publish("person/duration", json.dumps({"duration": time.time() - p_start}))

            # when new person enters
            if current_count > prev_count:
                total_persons += current_count - prev_count
                p_start = time.time()
            
            prev_count = current_count
            
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", json.dumps({"count": current_count,"total": total_persons}))
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_image)
        sys.stdout.buffer.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite("output_frame.png", out_image)
    # release resources
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()
    del infer_network


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
    sys.exit()
