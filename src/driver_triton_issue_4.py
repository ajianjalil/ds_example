#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
from pathlib import Path
import gi
import configparser
import argparse
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.FPS import PERF_DATA

import os
import pyds

import pickle
import socket
from queue import Queue
import threading
import numpy as np

import logging
import traceback

import json

logging.basicConfig(
    filename='app.log',      # Specify the file name
    filemode='a',            # Append mode
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Log format
    level=logging.INFO       # Set the logging level
)

# Get a logger instance
FILE_LOGGER = logging.getLogger('FILE_LOGGER')

no_display = False
silent = False
file_loop = True
perf_data = None

MAX_DISPLAY_LEN=64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]


MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]


CLASS_NB = 91
ACCURACY_ALL_CLASS = 0.5
UNTRACKED_OBJECT_ID = 0xffffffffffffffff
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
MIN_BOX_WIDTH = 32
MIN_BOX_HEIGHT = 32
TOP_K = 20
IOU_THRESHOLD = 0.3
OUTPUT_VIDEO_NAME = "./out.mp4"
MUXER_BATCH_TIMEOUT_USEC = 33000

MAX_NUM_SOURCES = 3

Counter=0


class TCPConnectionThread(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.queue = Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.sock = None

    def connect(self):
        while not self.stop_event.is_set():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                return True
            except Exception as e:
                print("From DS-7 pipeline- Error connecting:", e)
                time.sleep(1)  # Retry after 1 second
                traceback.print_exc()

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
                traceback.print_exc()
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
            traceback.print_exc()
            
            


def add_overlay_meta_to_frame(frame_object, batch_meta, frame_meta, label_names):
    """ Inserts an object into the metadata """
    
    # this is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution.
    rect_params = obj_meta.rect_params

    rect_params.left = int(frame_object.left)
    rect_params.top = int(frame_object.top)
    rect_params.width = int(frame_object.width)
    rect_params.height = int(frame_object.height)

    # FILE_LOGGER.info(rect_params.height)

    # Semi-transparent yellow backgroud
    rect_params.has_bg_color = 0
    rect_params.bg_color.set(1, 1, 0, 0.4)

    # Red border of width 3
    rect_params.border_width = 3
    rect_params.border_color.set(1, 0, 0, 1)

    # There is no tracking ID upon detection. The tracker will
    # assign an ID.
    UNTRACKED_OBJECT_ID = 0xffffffffffffffff
    obj_meta.object_id = UNTRACKED_OBJECT_ID

    # lbl_id = frame_object.classId
    # if lbl_id >= len(label_names):
    #     lbl_id = 0
    lbl_id=2 # constant
    # Set the object classification label.
    obj_meta.obj_label = label_names[lbl_id]

    """# Set display text for the object.
    txt_params = obj_meta.text_params
    if txt_params.display_text:
        pyds.free_buffer(txt_params.display_text)

    txt_params.x_offset = max(0,int(rect_params.left))
    txt_params.y_offset = max(0, int(rect_params.top) - 10)
    txt_params.display_text = "car"
    # Font , font-color and font-size
    txt_params.font_params.font_name = "Serif"
    txt_params.font_params.font_size = 10
    # set(red, green, blue, alpha); set to White
    txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    # Text background color
    txt_params.set_bg_clr = 1
    # set(red, green, blue, alpha); set to Black
    txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)"""
    
    # Inser the object into current frame meta
    # This object has no parent
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)



def tiler_sink_pad_buffer_probe(pad, info, u_data,connection_thread):

    global lock,condition,Counter,T000,global_overlays
    Counter=Counter+1
    # if Counter%100==0:
    #     print("probe running to read Ramis dict")
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        FILE_LOGGER.info("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    T0=time.time()
    # wait for lock2 for release
        # lock1 active
    source_id_list=[]        
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        source_id = frame_meta.source_id
        source_id_list.append(source_id)
        

        # try:
        #     # Create dummy owner object to keep memory for the image array alive
        #     owner = None
        #     # Getting Image data using nvbufsurface
        #     # the input should be address of buffer and batch_id
        #     # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
        #     data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(gst_buffer), frame_meta.batch_id)
        #     # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
        #     ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        #     ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        #     # Get pointer to buffer and create UnownedMemory object from the gpu buffer
        #     c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
        #     unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
        #     # Create MemoryPointer object from unownedmem, at index 0
        #     memptr = cp.cuda.MemoryPointer(unownedmem, 0)
        #     # Create cupy array to access the image data. This array is in GPU buffer
        #     #X=cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C').copy()
        #     #n_frame_gpu_batch[:,:,:,source_id] = X.view()
        #     FOR_RAMI[source_id] = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C').copy()[:,:,:3,cp.newaxis]
        # except:
        #     FILE_LOGGER.error(f"ERROR with source_id={source_id}",exc_info=True)
        #     traceback.print_exc()
        #     print(f"ERROR with source_id={source_id}")
        # with stream:
        #     #n_frame_gpu[:, :, 0] = 0.5 * n_frame_gpu[:, :, 0] + 0.5
        #     n_frame_gpu_batch[300:1000,300:1000, :] =0
        # stream.synchronize()
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    connection_thread.send_message(np.array(source_id_list))        
    return Gst.PadProbeReturn.OK




# frame_duration, number_frames = 1.0 / 30 * Gst.SECOND,0
def nvdsosd_sink_pad_buffer_probe(pad, info, u_data):

    global lock,condition,Counter,T000,global_overlays
    Counter=Counter+1

    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        FILE_LOGGER.info("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    T0=time.time()
    # wait for lock2 for release
        # lock1 active
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number = frame_meta.frame_num
        # global frame_duration

        # timestamp = int(frame_number * frame_duration)
        # gst_buffer.pts = gst_buffer.dts = int(timestamp)
        # gst_buffer.offset = timestamp

        l_obj=frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        source_id = frame_meta.source_id

        # cam_idx = source_id
        # overlay_index = to_python_index(source_id)
        # print(f"overlay_index={overlay_index} = to_python_index({source_id})")

        #=============start overlays=========================
        label_names=["Person","Vehicle","UnknownObject"]
        
        if 0 in global_overlays:
            # count = global_overlays["count"]
            # print(f"count={count}")
            # print(global_overlays)
            
            # polylines, rects, circles,polygons = decode_overlays(global_overlays[0])
            # Save the white screen image
            # if overlay_index==0:
            #     draw_overlays_on_white_screen(global_overlays[overlay_index], (1080,1920,3), f'./cvimg_with_overlay_from_deepstream/Idisp_with_overlays_{frame_number}.jpg')

            if True :    # poly lines will be always > 0 for camera with overlays
                try:
                    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    display_meta2 = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                    line_params = display_meta.line_params
                    # rect_params = display_meta.rect_params
                    line_params2 = display_meta2.line_params
                    line_index = 0
                    rect_index = 0
                    line_index2 = 0
                    circle_index = 0

                    display_meta.num_labels = 1  # Number of text elements
                    # Draw text
                    text_params = display_meta.text_params[0]
                    text_params.display_text = f"{global_overlays.get(0, 'none')}"  # Text to display
                    # FILE_LOGGER.info(pyds.get_string(text_params.display_text))

                    # Set text position
                    text_params.x_offset = 10  # X-coordinate for the text
                    text_params.y_offset = 10 # Y-coordinate for the text

                    # Set text font properties
                    text_params.font_params.font_name = "Serif"  # Font type
                    text_params.font_params.font_size = 20  # Font size
                    if frame_number%5 in [1,2,3]:
                        text_params.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)  # Yellow color with full opacity
                    else:
                        text_params.font_params.font_color.set(1.0, 0.0, 0.0, 1.0)  # Yellow color with full opacity

                    # Set text background color
                    text_params.set_bg_clr = 1  # Enable background color
                    text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)  # Black background with 50% transparency
                except:
                    # pprint(( circles ))
                    traceback.print_exc()
                    FILE_LOGGER.error("Error in adding overlays",exc_info=True)

                
                #==================End::multi overlays==================



            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta2)


        #=============end overlays==================



        obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
        }
        while l_obj is not None:
            try: 
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK





def get_label_names_from_file(filepath):
    """ Read a label file and convert it to string list """
    labels = "connected components"
    return labels





def make_element(element_name, i):
    """
    Creates a Gstreamer element with unique name
    Unique name is created by adding element type and index e.g. `element_name-i`
    Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
    :param element_name: The name of the element to create
    :param i: the index of the element in the pipeline
    :return: A Gst.Element object
    """
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element




def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")



def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "src" in name:
        source_element = child_proxy.get_by_name("src")
        if source_element:
            if source_element.find_property('drop-on-latency') != None:
                source_element.set_property("drop-on-latency", "true")
                source_element.set_property("latency", 400)
                # source_element.set_property("protocols", "udp")
                source_element.set_property("timeout", 50)
                source_element.set_property("tcp-timeout", 5000000)
                source_element.set_property("retry", 100)



def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    if file_loop:
        # use nvurisrcbin to enable file-loop
        uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("rtsp-reconnect-interval", 10)
        uri_decode_bin.set_property("rtsp-reconnect-attempts", 2160)
        # uri_decode_bin.set_property("cudadec-memtype", 2)

    else:
        uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def kill_the_pipeline(loop,pipeline,connection_thread):
    pipeline.set_state(Gst.State.NULL)
    connection_thread.stop()
    loop.quit()
    sys.exit(1)                    


def bus_call(bus, message, loop,pipeline,connection_thread):
    """
    This is call back from gstreamer
    
    This method takes 1 parameter and access the messages that is posted on pipeline's bus.
    
    :param loop: this is added from mainloop so that u can kill the loop to kill entire pipeline from messges
    :return: None
    """

    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        FILE_LOGGER.info("I am here at the EOS part")
        FILE_LOGGER.error(f"The EOS recieved from the triton Pipeline, Exiting the application")
        kill_the_pipeline(loop,pipeline,connection_thread)


    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        FILE_LOGGER.warning(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:

        err, debug = message.parse_error()
        FILE_LOGGER.error(f"Error: {err}: {debug}")
        print(f"Error: {err}: {debug}")
        src_element = message.src
        parent_obj = src_element.get_parent()
        if parent_obj:
            try:
                parent_obj_name = parent_obj.name
                FILE_LOGGER.info(f"parent object name is {parent_obj_name}")
                print(f"parent object name is {parent_obj.name}")
                # kill_the_pipeline(loop,pipeline,connection_thread)
            except Exception as e:
                FILE_LOGGER.info(traceback.print_exc())
                print("Error! exiting the app ")
                FILE_LOGGER.error("Error! exiting the app ",exc_info=True)
                # kill_the_pipeline(loop,pipeline,connection_thread)
    

    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        #Check for stream-eos message
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                #Set eos status of stream to True, to be deleted in delete-sources
                print("Got EOS from stream %d" % stream_id)
                FILE_LOGGER.info("Got EOS from stream %d" % stream_id)
                # Uncomment the below code if you need to restart the entire system
                # kill_the_pipeline(loop,pipeline,connection_thread) 


    return True




def receive_thread_loop(host, port, queue):
    global global_overlays
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
                        global_overlays = unpickled_obj
                        # print(f"time and count from overlay={global_overlays['time']},{global_overlays['count']}")
                        # print(global_overlays)
                        # queue.put(unpickled_obj)

            except socket.timeout:
                print("Cleaning up...socket thread from Triton module:: Timeout while waiting for connection, ")
                time.sleep(1)
            except EOFError:
                print("EOF Error")



import queue
overlay_queue = queue.Queue(maxsize=15)

def main(args, requested_pgie=None, config=None, disable_probe=False):
    global overlay_queue
    host = '127.0.0.1'
    port = 54321     

    number_sources=len(args)

    # connection_thread = TCPConnectionThread(host, port)
    # connection_thread.start()
    #connection_thread.send_message(stats)

    global perf_data
    perf_data = PERF_DATA(len(args))

    

    # Standard GStreamer initialization
    Gst.init(None)




    host = '127.0.0.1'
    port = 54321     
    connection_thread = TCPConnectionThread(host, port)
    connection_thread.start()

    receive_thread = threading.Thread(target=receive_thread_loop, args=("127.0.0.1", 9876, overlay_queue))
    receive_thread.daemon = True
    receive_thread.start()       

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    is_live = True
    for i in range(number_sources):
        print("Creating source_bin ", i, "\n")
        uri_name = args[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)

        # Create nvvideoconvert, videorate, and caps filter elements
        nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert_%u" % i)
        if not nvvideoconvert:
            sys.stderr.write("Unable to create nvvideoconvert \n")
        
        videorate = Gst.ElementFactory.make("videorate", "videorate_%u" % i)
        if not videorate:
            sys.stderr.write("Unable to create videorate \n")
        
        caps_filter = Gst.ElementFactory.make("capsfilter", "capsfilter_%u" % i)
        if not caps_filter:
            sys.stderr.write("Unable to create caps filter \n")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=25/1")
        caps_filter.set_property("caps", caps)

        # Add elements to the pipeline
        pipeline.add(nvvideoconvert)
        pipeline.add(videorate)
        pipeline.add(caps_filter)

        # Link elements: source_bin -> nvvideoconvert -> videorate -> caps_filter -> streammux
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        sinkpad = nvvideoconvert.get_static_pad("sink")
        if not sinkpad:
            sys.stderr.write("Unable to get sink pad for nvvideoconvert \n")
        srcpad.link(sinkpad)

        nvvideoconvert.link(videorate)
        videorate.link(caps_filter)

        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = caps_filter.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)

    nvdslogger = None




    print("Creating Pgie \n ")
    if True :
        pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    # elif requested_pgie != None and requested_pgie == 'nvinfer':
    #     pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    # else:
    #     pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")

    if not pgie:
        sys.stderr.write(" Unable to create pgie :  %s\n" % requested_pgie)

    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad ")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0,connection_thread)


    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)

    if file_loop:
        if is_aarch64():
            # Set nvbuf-memory-type=4 for aarch64 for file-loop (nvurisrcbin case)
            streammux.set_property('nvbuf-memory-type', 4)
        else:
            # Set nvbuf-memory-type=2 for x86 for file-loop (nvurisrcbin case)
            # streammux.set_property('nvbuf-memory-type', 1)
            pass

    if no_display:
        print("Creating Fakesink \n")
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
        sink.set_property('enable-last-sample', 0)
        sink.set_property('sync', 0)
    else:
        if is_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
            if not sink:
                sys.stderr.write(" Unable to create egl sink \n")

    if not sink:
        sys.stderr.write(" Unable to create sink element \n")

    # if is_live:
    #     print("At least one of the sources is live")
    #     streammux.set_property('live-source', 1)

    if os.environ["USE_NEW_NVSTREAMMUX"]=="yes":
        streammux.set_property("config-file-path", "/opt/nvidia/deepstream/deepstream-7.0/sources/src/streammux.txt")
        streammux.set_property("batch-size", 30)
        print("###################Using the new SteramMUX########################\n\n\n\n")
    else:
        streammux.set_property("batched-push-timeout", 25000)
        streammux.set_property("width", 640) #640 3840
        streammux.set_property("height", 480) #480 2160
        streammux.set_property("batch-size", 30)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    if requested_pgie == "nvinferserver" and config != None:
        pgie.set_property('config-file-path', config)
    elif requested_pgie == "nvinferserver-grpc" and config != None:
        pgie.set_property('config-file-path', config)
    elif requested_pgie == "nvinfer" and config != None:
        pgie.set_property('config-file-path', config)
    else:
        # pgie.set_property('config-file-path', "/opt/nvidia/deepstream/deepstream-7.0/sources/src/dstest1_pgie_inferserver_config_fake_1080.txt") # dstest1_pgie_inferserver_config_torch_resnet50.txt
        # pgie.set_property('config-file-path', "/opt/nvidia/deepstream/deepstream-7.0/sources/src/dstest1_pgie_inferserver_config_fake_640_10.txt") # dstest1_pgie_inferserver_config_torch_resnet50.txt dstest1_pgie_inferserver_config_fake_1080_10.txt
        pgie.set_property("config-file-path","/opt/nvidia/deepstream/deepstream-7.0/sources/src/dstest1_pgie_inferserver_config_fake_1080_10.txt")


        print('########################################OK')
        # pgie.set_property('config-file-path', "/opt/nvidia/deepstream/deepstream-7.0/sources/src/dstest1_pgie_inferserver_config_torch_resnet50.txt")
    # pgie_batch_size=pgie.get_property("batch-size")
    # if(pgie_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
    #     pgie.set_property("batch-size",number_sources)
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    if nvdslogger:
        pipeline.add(nvdslogger)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    # streammux.link(queue2)

    """    if nvdslogger:
        queue2.link(nvdslogger)
        nvdslogger.link(tiler)
    else:
        queue2.link(tiler)
    tiler.link(queue3)
    queue3.link(nvvidconv)
    nvvidconv.link(queue4)
    queue4.link(nvosd)
    nvosd.link(queue5)
    queue5.link(sink)  """ 

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop,pipeline,connection_thread)
    # pgie_src_pad=pgie.get_static_pad("src")
    # if not pgie_src_pad:
    #     sys.stderr.write(" Unable to get src pad \n")
    # else:
    #     if not disable_probe:
    #         pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0,connection_thread)
            # perf callback function to print fps every 5 sec
            # GLib.timeout_add(5000, perf_data.perf_print_callback)

    #===========================================The new Code==========================================================
    nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
    if not nvstreamdemux:
        sys.stderr.write(" Unable to create nvstreamdemux ")
    

    # self.pipeline.add(pgie)
    pipeline.add(nvstreamdemux)
    queue2.link(nvstreamdemux)

    debug_auto_videosink = False





    for i in range(len(stream_paths)):
        # pipeline nvstreamdemux -> queue -> nvvidconv -> nvosd -> (if Jetson) nvegltransform -> nveglgl
        # Creating EGLsink
        # creating queue
        queue = make_element("queue", i)
        pipeline.add(queue)

        # creating nvvidconv
        nvvideoconvert = make_element("nvvideoconvert", i)
        pipeline.add(nvvideoconvert)

        # creating nvosd
        nvdsosd = make_element("nvdsosd", i)
        pipeline.add(nvdsosd)
        nvdsosd.set_property("process-mode", OSD_PROCESS_MODE)
        nvdsosd.set_property("display-text", OSD_DISPLAY_TEXT)

        # connect nvstreamdemux -> queue
        padname = "src_%u" % i
        demuxsrcpad = nvstreamdemux.get_request_pad(padname)
        if not demuxsrcpad:
            sys.stderr.write("Unable to create demux src pad ")

        queuesinkpad = queue.get_static_pad("sink")
        if not queuesinkpad:
            sys.stderr.write("Unable to create queue sink pad ")
        demuxsrcpad.link(queuesinkpad)


        # connect  queue -> nvvidconv -> nvosd -> nveglgl
        queue.link(nvvideoconvert)
        nvvideoconvert.link(nvdsosd)

        nvds_sink_pad = nvdsosd.get_static_pad("sink")
        if not nvds_sink_pad:
            sys.stderr.write(" Unable to get nvdsosd sink pad ")
        else:
            nvds_sink_pad.add_probe(Gst.PadProbeType.BUFFER, nvdsosd_sink_pad_buffer_probe, 0)

        tee = make_element("tee", i)
        pipeline.add(tee)
        
        nvdsosd.link(tee)
        nvdosd_branch_pad = tee.get_request_pad("src_%u")
        if not nvdosd_branch_pad:
            sys.stderr.write("Unable to create tee src pad ")

        if debug_auto_videosink:
            autovideosink = make_element("autovideosink", i)
            autovideosink.set_property("sync","false")
            pipeline.add(autovideosink)
            

            nvconvert_branch_pad = tee.get_request_pad("src_%u")
            if not nvconvert_branch_pad:
                sys.stderr.write("Unable to create nvconvert_branch_pad tee src pad ")


            nvvidconv_autovideosink = Gst.ElementFactory.make("nvvideoconvert", f"convertor_autovideosink{i}")
            if not nvvidconv_autovideosink:
                sys.stderr.write("Unable to create nvvideoconvert for autovideosink")

            pipeline.add(nvvidconv_autovideosink)



            nvvidconv_sink_pad = nvvidconv_autovideosink.get_static_pad("sink")
            nvconvert_branch_pad.link(nvvidconv_sink_pad)


            autovideosink_sink_pad = autovideosink.get_static_pad("sink")
            nvvidconv_autovideosink_src_pad = nvvidconv_autovideosink.get_static_pad("src")


            nvvidconv_autovideosink_src_pad.link(autovideosink_sink_pad)



        nvvidconv_postosd = Gst.ElementFactory.make(
            "nvvideoconvert", f"convertor_postosd{i}")
        if not nvvidconv_postosd:
            sys.stderr.write(" Unable to create nvvidconv_postosd ")



        # Create a caps filter
        caps = Gst.ElementFactory.make("capsfilter", f"filter{i}")
        caps.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
        )

        nvvidconv_postosd.link(caps)

        encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder{i}")
        encoder.set_property("force-intra", "true")
        encoder.set_property("iframeinterval", 25)
        encoder.set_property("force-idr", "true")
        encoder.set_property("idrinterval", 25)            
        encoder.set_property("control-rate", 1)

        print("Creating H264 Encoder")

        if not encoder:
            sys.stderr.write(" Unable to create encoder")

        if is_aarch64():
            encoder.set_property("preset-level", 1)
            encoder.set_property("insert-sps-pps", 1)
            #encoder.set_property("bufapi-version", 1)

        sink = Gst.ElementFactory.make("rtspclientsink", f"rtspclientsink{i}")
        if not sink:
            sys.stderr.write(" Unable to create udpsink")
        sink.set_property("location",f"rtsp://127.0.0.1:554/video{i}")
        sink.set_property("protocols","tcp")
        print(f"will stream in rtsp://127.0.0.1:554/video{i}")
        pipeline.add(nvvidconv_postosd)
        pipeline.add(caps)
        pipeline.add(encoder)
        pipeline.add(sink)

        # nvdsosd.link(nvvidconv_postosd)


        nvvidconv_postosd_sink = nvvidconv_postosd.get_static_pad("sink")
        nvdosd_branch_pad.link(nvvidconv_postosd_sink)

        nvvidconv_postosd.link(caps)
        caps.link(encoder)
        encoder.link(sink)

    #===========================================The end================================================================


    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    connection_thread.stop()
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def parse_args():

    
    global no_display
    global silent
    global file_loop
    pgie = None
    config = None
    disable_probe = False
    # no_display = args.no_display
    # silent = args.silent
    # file_loop = args.file_loop

    # if config and not pgie or pgie and not config:
    #     sys.stderr.write ("\nEither pgie or configfile is missing. Please specify both! Exiting...\n\n\n\n")
    #     parser.print_help()
    #     sys.exit(1)
    # if config:
    #     config_path = Path(config)
    #     if not config_path.is_file():
    #         sys.stderr.write ("Specified config-file: %s doesn't exist. Exiting...\n\n" % config)
    #         sys.exit(1)

    # print(vars(args))
    return pgie, config, disable_probe


if __name__ == '__main__':


    pgie, config, disable_probe = parse_args()
    # PLease add logging statements of the following content in the below list
    stream_paths = ['rtsp://192.168.10.135:8555/video1','rtsp://192.168.10.135:8555/video1']
                    # 'rtsp://192.168.10.135:8555/video1']
    # stream_paths = ['rtsp://192.168.43.189:8555/video1',
    #                 'rtsp://192.168.43.189:8555/video1']
    # stream_paths = ['rtsp://192.168.1.8:8555/video1',
    #                 'rtsp://localhost:8554/stream']   
    # stream_paths = ['rtsp://localhost:8554/stream']
    sys.exit(main(stream_paths, pgie, config, disable_probe))

