
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import json
import cv2
import numpy as np
import triton_python_backend_utils as pb_utils
import socket
from queue import Queue
import pickle
import threading
import time
import queue
import cupy as cp
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import traceback
import tensorflow as tf

from pprint import pprint

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.



def receive_thread(host, port, queue):
    # Create a TCP socket and listen for incoming connections
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Enable TCP Keep-Alive
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Set the idle time before sending the first keep-alive packet (in seconds)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)  # Adjust as needed
        # Set the interval between subsequent keep-alive packets (in seconds)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)  # Adjust as needed
        # Set the number of failed keep-alive probes before considering the connection dead
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5)  
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Set the reuse address option
        s.settimeout(0.01)  # Timeout set to 10ms
        s.bind((host, port))
        s.listen()
        print("Waiting for connection...")
        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    while True:
                        # Receive the length of the data
                        length_bytes = conn.recv(4)
                        if not length_bytes:
                            break  # If no data received, exit loop
                        # Convert length bytes to integer
                        data_length = int.from_bytes(length_bytes, byteorder='big')
                        # Receive the serialized data
                        serialized_data = b''
                        while len(serialized_data) < data_length:
                            packet = conn.recv(data_length - len(serialized_data))
                            if not packet:
                                break
                            serialized_data += packet
                        if len(serialized_data) != data_length:
                            print("Incomplete data received")
                            continue
                        # Unpickle the object
                        unpickled_obj = pickle.loads(serialized_data)
                        # Put the received object into the queue
                        # print(unpickled_obj)
                        queue.put(unpickled_obj)

            except socket.timeout:
                print("Cleaning up...socket thread from Triton module:: Timeout while waiting for connection, ")
                time.sleep(1)
            except EOFError:
                print("EOF Error")



# additions





class TCPConnectionThread(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.sock = None

    def connect(self):
        while not self.stop_event.is_set():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                return True
            except Exception as e:
                print("From DS-7 Triton inference server-Error connecting:", e)
                time.sleep(1)  # Retry after 1 second

        return False

    def run(self):
        while not self.stop_event.is_set():
            if not self.sock:
                if not self.connect():
                    # If unable to connect, wait before retrying
                    time.sleep(1)
                    continue

            try:
                while not self.stop_event.is_set():
                    if not self.queue.empty():
                        data = self.queue.get()
                        # Pickle the data
                        serialized_data = pickle.dumps(data)
                        # Calculate the length of the data
                        data_length = len(serialized_data)
                        # Convert data length to bytes (4 bytes for an integer)
                        length_bytes = data_length.to_bytes(4, byteorder='big')
                        # Send the length of the data
                        self.sock.sendall(length_bytes)
                        # Send the data
                        self.sock.sendall(serialized_data)
                    else:
                        time.sleep(0.1)  # Sleep briefly to avoid busy waiting
            except Exception as e:
                print("Error sending data:", e)
                self.sock.close()
                self.sock = None

    def stop(self):
        self.stop_event.set()
        if self.sock:
            self.sock.close()

    def send_message(self, obj):
        try:
            self.queue.put(obj,block=False)
        except Exception as e:
            pass
            # print(e)
    
import time

def to_cam_index(List_of_Cameras_indexes,python_index):
    """Convert Python index to natural number index (camera index)."""
    if 0 <= python_index < len(List_of_Cameras_indexes):
        return List_of_Cameras_indexes[python_index]
    return None

def to_python_index(cam_idx_to_python_idx_map, cam_index):
    """Convert natural number index (camera index) to Python index."""
    return cam_idx_to_python_idx_map.get(cam_index, None)


import pickle


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """


    def initialize(self, args):
        self.AAA={}
        self.MAX_MEM=0

        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        
        """
        resolution=list((480,640,3))
        MODEL_PATH = './efficientnet-tensorflow2-b0-classification-v1'
        self.detect_fn = tf.saved_model.load(MODEL_PATH)

        # Start the TCP connection thread
        # host = "127.0.0.1"  # Change to your server's IP
        # port = 9876         # Change to your server's port

        # self.tcp_thread = TCPConnectionThread(host, port)
        # self.tcp_thread.start()
        # self.source_id_q = Queue(maxsize=100)
        # self.receive_thread = threading.Thread(target=receive_thread, args=("127.0.0.1", 54321, self.source_id_q))
        # self.receive_thread.start()  
        # print("TCP Connection Thread started")
        self.indexes_from_pipeline = []
        self.c = 0

    def find_order(self):
        try:
            self.indexes_from_pipeline = self.source_id_q.get(block=False)
            # print(f"source_id list from model = {indexes_from_pipeline}")
        except:
            print("index list from pipeline to triton is delayed, please wait")
            traceback.print_exc()
        return self.indexes_from_pipeline

    def generate_tiled_frame(self, batch_size, frames):
        # rows = int(np.ceil(batch_size / 5))
        # cols = min(5, batch_size)
        frames = cp.asnumpy(frames)
        tile_height, tile_width = 480, 1280
        tiled_frame = np.zeros((tile_height, tile_width , 3), dtype=np.uint8)
        #print(f"shape={frames.shape}")
        indexes_from_pipeline = []
        indexes_from_pipeline = self.indexes_from_pipeline
        
        tiled = []
        if batch_size==len(indexes_from_pipeline): #sometime it wont be equal, then skip
            for i in range(batch_size):
                frame = frames[i]
                if frame is not None:
                    # Resize the frame to fit the tile size
                    # frame = np.transpose(frame, (1, 2, 0))
                    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    text = f"index:{indexes_from_pipeline[i]}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2
                    tiled.append(cv2.putText(frame.astype(np.uint8), text, (int((frame.shape[1] - cv2.getTextSize(text, font, font_scale, font_thickness)[0][0]) / 2), int((frame.shape[0] + cv2.getTextSize(text, font, font_scale, font_thickness)[0][1]) / 2)), font, font_scale, (255, 255, 255), font_thickness))
                    if len(tiled)>=5:
                        break
            horizontal_concatenated = cv2.hconcat(tiled)
            tiled_frame = cv2.resize(horizontal_concatenated, (tile_width, tile_height))

        return tiled_frame,indexes_from_pipeline

    def execute(self, requests):

        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
       

        # Establish TCP socket connection
        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.sock.connect((self.tcp_host, self.tcp_port))
        # print("TCP socket connection established")

        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            # frame = cp.fromDlpack(input_tensor.to_dlpack())

            tensor = tf.experimental.dlpack.from_dlpack(input_tensor.to_dlpack())
            print(tensor.shape)
            # batch_size = frame.shape[0]
            # print(['HH00',frame.shape])
            # self.count+=1
            # index = self.count%2000
            # self.connection_thread.send_message(self.list_of_overlays[index])

            rects_for_all_channels = []
            batch_size = tensor.shape[0]


            # 2. Preprocessing (Optional: depends on your model; many accept uint8)
            # If your model needs float32 normalized to [0, 1]
            input_tensor_prepared = tf.cast(tensor, tf.float32) / 255.0
            # 1. Resize the input to 640x640
            input_resized = tf.image.resize(input_tensor_prepared, [224, 224])

            # 2. Convert back to tf.uint8
            input_resized_uint8 = tf.cast(input_resized * 255.0, tf.float32)

            # 3. Run detection
            detections = self.detect_fn(input_resized_uint8)
            # print(detections)


            # try:
            #     cv2.imshow("Tiler Debug Window from Triton", cp.asnumpy(frame)[0].astype(np.uint8))
            #     key = cv2.waitKey(1)
            # except Exception as e:
            #     print("Exception occurred")
            #     traceback.print_exc()

            rects_for_all_channels = np.zeros((batch_size, 15, 4))
            stats = rects_for_all_channels.astype(np.float32)
            shape = np.array([int(stats.shape[1]),int(stats.shape[2])])
            shape = np.tile(shape, (batch_size, 1,1)) # batch size
            shape = shape.astype(np.float32)
            out_tensor = pb_utils.Tensor("OUTPUT0", stats)
            out_tensor2 = pb_utils.Tensor("OUTPUT1", shape)
            responses.append(pb_utils.InferenceResponse([out_tensor,out_tensor2]))
            
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        # Close the TCP socket connection
        # self.sock.close()
        # self.connection_thread.stop()
        print("Cleaning up...")
