#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        """initialize all class variables as None"""
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model_xml, device, cpu_extension=None, num_requests=1):
        """
        Load model in IECore as ExecutableNetwork.

        Read model file and make IENetwork.
        Check for Supported Layers in IENetwork according to selected Devices i.e. CPU.
        Load IENetwork into IECore as ExecutableNetwork

        Parameters:
            model_xml: The xml file of our Model.
            device (string): Device you want to use to run inference.
            cpu_extension: The helper extension you want to use for CPU.
            num_requests (int): number of inference requests you want to perform

        Returns:
            ExecutableNetwork: Loaded model which is executable by IEngine.
        """
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ### TODO: Load the model ###
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=model_xml, weights=model_bin)
        
        ### TODO: Add any necessary extensions ###
        if cpu_extension and device == 'CPU':
            self.plugin.add_extension(extension_path=cpu_extension, device_name=device)
        
        ### TODO: Check for supported layers ###
        layers_map = self.plugin.query_network(network=self.network, device_name=device)
        
        unsupported_layers = [l for l in self.network.layers.keys() if l not in layers_map]
        if (unsupported_layers != []):
            sys.exit("Those mention layers in your model are not supported by OpenVino Inference Engine:" \
                     " \n\t" + "\n\t".join(unsupported_layers))
        
        # retrieve name of model's input layer
        self.input_blob = next(iter(self.network.inputs))
        
        # retrieve name of model's input layer
        self.output_blob = next(iter(self.network.outputs))
        
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return self.plugin.load_network(network=self.network, device_name=device, num_requests=num_requests)



    def get_input_shape(self):
        """Return the shape of the input layer"""
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape


    def exec_net(self, input_image):
        """Start an asynchronous request"""
        ### TODO: Start an asynchronous request ###
        self.infer_request = self.exec_network.start_async(request_id=0, inputs={self.input_blob: input_image})
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return None


    def wait(self):
        """Wait for the request to be complete."""
        
        ### TODO: Wait for the request to be complete. ###
        status = self.infer_request.wait(-1)
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status


    def get_output(self):
        """Extract and return the output results from inference request"""
        
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.infer_request.outputs[self.output_blob]

