#!/usr/bin/python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# from Rules_Reader import Master
import threading
import config
import multiprocessing as mp
import threading
import logging

from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pickle
import sys
if sys.platform == "linux" or sys.platform == "linux2": 
    import ctypes 
    ctypes.CDLL('libX11.so').XInitThreads()
sys.path.append('../')
import gi

gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import sys
import os
from common.platform_info import PlatformInfo
from common.FPS import PERF_DATA
import pyds
import math

import ctypes
import cupy as cp


perf_data = None

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str= ["Vehicle", "TwoWheeler", "Person","RoadSign"]



import numpy as np
import time

import queue
import traceback
import cv2
# import nvtx
from pprint import pprint
from itertools import chain
# Global variabels
FOR_RAMI={}
lock = threading.Lock()
condition = threading.Condition(lock)
cam_idx_to_python_idx_map = {}
cam_idx_to_source_bin_map = {}
cam_idx_to_uri_map = {}
to_recover_set = set()
pipeline = None
streammux = None
status_queue = queue.Queue(maxsize=15)
global_overlays={}

FILE_LOGGER = logging.getLogger("file_logger")
FILE_LOGGER.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
FILE_LOGGER.addHandler(handler)

BASE_ALARM_INDEX = 0

class CudaArrayInterface:
    def __init__(self, ptr, shape, dtype_str):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": dtype_str,
            "data": (ptr, False),  # False indicates no ownership of memory
            "stream": None  # Optional: Add CUDA stream if required
        }


def cv_cuda_gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    assert len(arr.shape) in (2, 3), "CuPy array must have 2 or 3 dimensions to be a valid GpuMat"
    type_map = {
        cp.dtype('uint8'): cv2.CV_8U,
        cp.dtype('int8'): cv2.CV_8S,
        cp.dtype('uint16'): cv2.CV_16U,
        cp.dtype('int16'): cv2.CV_16S,
        cp.dtype('int32'): cv2.CV_32S,
        cp.dtype('float32'): cv2.CV_32F,
        cp.dtype('float64'): cv2.CV_64F
    }
    depth = type_map.get(arr.dtype)
    assert depth is not None, "Unsupported CuPy array dtype"
    channels = 1 if len(arr.shape) == 2 else arr.shape[2]
    # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
    # (depth&7) + ((channels - 1) << 3)
    mat_type = depth + ((channels - 1) << 3)
    mat = cv2.cuda.createGpuMatFromCudaMemory(arr.__cuda_array_interface__['shape'][1::-1],
                                              mat_type,
                                              arr.__cuda_array_interface__['data'][0])
    return mat

def cp_array_from_cv_cuda_gpumat(mat: cv2.cuda.GpuMat) -> cp.ndarray:
    class CudaArrayInterface:
        def __init__(self, gpu_mat: cv2.cuda.GpuMat):
            w, h = gpu_mat.size()
            type_map = {
                cv2.CV_8U: "|u1",
                cv2.CV_8S: "|i1",
                cv2.CV_16U: "<u2", cv2.CV_16S: "<i2",
                cv2.CV_32S: "<i4",
                cv2.CV_32F: "<f4", cv2.CV_64F: "<f8",
            }
            self.__cuda_array_interface__ = {
                "version": 3,
                "shape": (h, w, gpu_mat.channels()) if gpu_mat.channels() > 1 else (h, w),
                "typestr": type_map[gpu_mat.depth()],
                "descr": [("", type_map[gpu_mat.depth()])],
                "stream": 1,
                "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()) if gpu_mat.channels() > 1
                else (gpu_mat.step, gpu_mat.elemSize()),
                "data": (gpu_mat.cudaPtr(), False),
            }
    arr = cp.asarray(CudaArrayInterface(mat))

    return arr




def run_connected_components_on_batch(Batch,key):
        # Transpose frame within GPU and convert to grayscale
        frame = Batch

        frame = frame.astype(cp.uint8)
        # Create GpuMat from CuPy array using zero-copy
        gpu_frame = cv_cuda_gpumat_from_cp_array(frame)
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGBA2GRAY)

        # # Apply Gaussian blur on GPU
        gaussian_filter = cv2.cuda.createGaussianFilter(gpu_gray.type(), gpu_gray.type(), (31, 31), 0)
        gpu_blurred = gaussian_filter.apply(gpu_gray)

        # Apply thresholding on the blurred image
        _, gpu_thresh = cv2.cuda.threshold(gpu_blurred, 127, 255, cv2.THRESH_BINARY)


        # Apply connected components on GPU
        labels = cv2.cuda.connectedComponents(gpu_thresh)
        labels_cp = cp_array_from_cv_cuda_gpumat(labels)


        num_labels = int(cp.asnumpy(labels_cp.max()))
        unique_labels = cp.unique(labels_cp)
        num_labels = len(unique_labels)
        
        # Generate random colors for each label
        label_colors = cp.random.randint(0, 256, (num_labels, 3), dtype=cp.uint8)
        
        # Create a color image based on the labels and colors
        color_image = label_colors[labels_cp.get()].reshape(labels_cp.shape + (3,))

        # self.Batch[:, :, :, i] = color_image
        # Batch[:] = color_image[:]

        # new code
        bounding_boxes=[]
        rects=[]
        for label in unique_labels:

            # Create a binary mask for the current label
            component_mask = (labels_cp == label).astype(cp.uint8)
            # Find the indices of all points belonging to the component
            # Find the indices of all points belonging to the component
            y_indices, x_indices = cp.where(component_mask)
            
            # Calculate the bounding box if the component has points
            if y_indices.size > 0 and x_indices.size > 0:
                x_min = cp.min(x_indices)
                x_max = cp.max(x_indices)
                y_min = cp.min(y_indices)
                y_max = cp.max(y_indices)
                rects.append(['rectangle', ((x_min, y_min), (x_max,y_max), (0, 205, 0), 2)])
                
                # Store the bounding box as (x_min, y_min, x_max, y_max)
                bounding_box = (x_min.item(), y_min.item(), x_max.item(), y_max.item())
                bounding_boxes.append(bounding_box)
            else:
                # Handle case where the component is empty
                bounding_boxes.append(None)



def find_alarm_id_based_on_quadrant(Alarms_OUTS):
    global BASE_ALARM_INDEX
    if Alarms_OUTS[6] == Alarms_OUTS[7]:
        alarm_id = Alarms_OUTS[6]
    else:
        alarm_id = Alarms_OUTS[BASE_ALARM_INDEX]
    return alarm_id


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

def to_cam_index(List_of_Cameras_indexes,python_index):
    """Convert Python index to natural number index (camera index)."""
    if 0 <= python_index < len(List_of_Cameras_indexes):
        return List_of_Cameras_indexes[python_index]
    return None

def to_python_index( cam_index):
    global cam_idx_to_python_idx_map
    """Convert natural number index (camera index) to Python index."""
    return cam_idx_to_python_idx_map.get(cam_index, None)

def M(most_common_resolution,List_of_Cameras_indexes):
    global global_overlays
    Master = None
    NUM = len(List_of_Cameras_indexes)
    width,height = most_common_resolution
    Batch=cp.zeros((height,width,3,NUM))
    while True:       
        for python_idx in range(0,NUM):
            Batch[:,:,:,python_idx:(python_idx+1)]=FOR_RAMI[to_cam_index(List_of_Cameras_indexes,python_idx)]

            # Extract a single frame from the batch for processing
            frame = Batch[:, :, :, python_idx].copy()
            # Perform operations on the single frame if needed
            frame = frame.astype(cp.uint8)
            # Create GpuMat from CuPy array using zero-copy
            gpu_frame = cv_cuda_gpumat_from_cp_array(frame)
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2GRAY)

            # # Apply Gaussian blur on GPU
            gaussian_filter = cv2.cuda.createGaussianFilter(gpu_gray.type(), gpu_gray.type(), (31, 31), 0)
            gpu_blurred = gaussian_filter.apply(gpu_gray)

            # Apply thresholding on the blurred image
            _, gpu_thresh = cv2.cuda.threshold(gpu_blurred, 127, 255, cv2.THRESH_BINARY)


            # Apply connected components on GPU
            labels = cv2.cuda.connectedComponents(gpu_thresh)
            labels_cp = cp_array_from_cv_cuda_gpumat(labels)


            num_labels = int(cp.asnumpy(labels_cp.max()))
            unique_labels = cp.unique(labels_cp)
            num_labels = len(unique_labels)
            
            # Generate random colors for each label
            label_colors = cp.random.randint(0, 256, (num_labels, 3), dtype=cp.uint8)
            
            # Create a color image based on the labels and colors
            color_image = label_colors[labels_cp.get()].reshape(labels_cp.shape + (3,))
            # Simulate high GPU usage by performing a large matrix multiplication
            a = cp.random.rand(*Batch.shape)
            result = cp.einsum('ijkl,ijkl->ijkl', Batch, a)
            # Additional operations to increase GPU utilization
            b = cp.random.rand(*Batch.shape)
            result += cp.einsum('ijkl,ijkl->ijkl', Batch, b)
            print(cp.sum(Batch))
            random_delay = cp.random.uniform(0.001, 0.02).item()
            time.sleep(random_delay)
        # print("Batch shape",Batch.shape)
            

Counter=0
T000=time.time()


def decode_overlays(overlay_data):
    # for overlay_item in overlay_data:
    polylines = []
    rects = []
    circles = []
    polygons = []
    # pprint(overlay_data)
    try:
        polylines_data = overlay_data['Data']
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
        for data_item in polylines_data:
            if data_item[0] == 'polylines':
                polyline_coords = data_item[1][0]
                is_polygon = True if len(data_item) ==2 else False
                # pprint(data_item)
                if is_polygon:
                    polyline_array = np.array(polyline_coords, dtype=np.int32)
                    polyline_array = polyline_array.reshape(-1, 2)
                    polygons.append(polyline_array)
                else:
                    # print(polyline_coords)
                    polyline_array = np.array(polyline_coords, dtype=np.int32)
                    polyline_array = polyline_array.reshape(-1, 2)
                    # print(f"polyline_array={polyline_array}")
                    # FILE_LOGGER.info("Polyline points:", polyline_array)
                    polylines.append(polyline_array)
            elif data_item[0] == 'rectangle':
                try:
                    # FILE_LOGGER.info(f"Rectangle points={data_item[1]}")
                    rect = [
                        data_item[1][0][0],  # x1
                        data_item[1][0][1],  # y1
                        data_item[1][1][0],  # x2
                        data_item[1][1][1]   # y2
                    ]
                    rects.append(rect)
                except Exception as e:
                    FILE_LOGGER.info("decode overlay in rectangle failure",exc_info=True)
            elif data_item[0] == 'circle':
                try:
                    circle = [
                        data_item[1][0],  # x
                        data_item[1][1],  # y
                        data_item[1][2]   # radius
                    ]
                    circles.append(circle)
                except Exception as e:
                    FILE_LOGGER.info("decode overlay in circle failure",exc_info=True)
    except Exception as e:
        # print(overlay_data)
        traceback.print_exc()
        FILE_LOGGER.info("decode overlay failure",exc_info=True)
    return polylines, rects, circles, polygons



def draw_overlays_on_white_screen(overlay_data, image_shape, output_path):
    """
    Draw overlays on a white screen.

    Args:
        overlay_data (dict): Overlay data to decode and draw.
        image_shape (tuple): Shape of the white screen (height, width, channels).
        output_path (str): Path to save the resulting image.
    """
    # Decode the overlays
    polylines, rects, circles = decode_overlays(overlay_data)
    
    # Create a white screen
    white_screen = np.full(image_shape, 255, dtype=np.uint8)
    
    # Draw rectangles
    for rect in rects:
        x1, y1, x2, y2 = rect
        cv2.rectangle(white_screen, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)  # Red rectangles
    
    # (Optional) Draw polylines, if needed
    for polyline in polylines:
        cv2.polylines(white_screen, [polyline], isClosed=True, color=(0, 255, 0), thickness=2)  # Green polylines

    # Save the resulting image
    # cv2.imwrite(output_path, white_screen)
    # print(f"Image saved with overlays at {output_path}")


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



def tiler_sink_pad_buffer_probe(pad, info, u_data):

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

        try:
            # Create dummy owner object to keep memory for the image array alive
            owner = None
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
            data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(gst_buffer), frame_meta.batch_id)
            # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            # Get pointer to buffer and create UnownedMemory object from the gpu buffer
            c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
            unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
            # Create MemoryPointer object from unownedmem, at index 0
            memptr = cp.cuda.MemoryPointer(unownedmem, 0)
            # Create cupy array to access the image data. This array is in GPU buffer
            #X=cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C').copy()
            #n_frame_gpu_batch[:,:,:,source_id] = X.view()
            FOR_RAMI[source_id] = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C').copy()[:,:,:3,cp.newaxis]
        except:
            FILE_LOGGER.error(f"ERROR with source_id={source_id}",exc_info=True)
            traceback.print_exc()
            print(f"ERROR with source_id={source_id}")
        # with stream:
        #     #n_frame_gpu[:, :, 0] = 0.5 * n_frame_gpu[:, :, 0] + 0.5
        #     n_frame_gpu_batch[300:1000,300:1000, :] =0
        # stream.synchronize()
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

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
        overlay_index = to_python_index(source_id)

        #=============start overlays=========================
        label_names=["Person","Vehicle","UnknownObject"]
        
        if overlay_index in global_overlays:
            polylines, rects, circles,polygons = decode_overlays(global_overlays[overlay_index])
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

                    # print(f"len of polylines={len(polylines)}")
                    # for polyline in polylines:
                        
                        
                    #     sampled_points = sample_polyline_points(polyline, max_points=16,is_polygon=False)
                    #     # sampled_points = polyline
                    #     # print(f"size and content={sampled_points}")
                    #     for i in range(len(sampled_points) - 1):
                    #         x1, y1 = sampled_points[i]
                    #         x2, y2 = sampled_points[i + 1]
                            

                    #         if line_index2 < 16:   # deepstream only supports 16 lines
                    #             # print(f"line_index={line_index}")
                    #             line_params2[line_index2].x1 = x1
                    #             line_params2[line_index2].y1 = y1
                    #             line_params2[line_index2].x2 = x2
                    #             line_params2[line_index2].y2 = y2
                    #             line_params2[line_index2].line_color.set(1.0, 0.0, 0.5, 1.0)  # Green color
                    #             line_params2[line_index2].line_width = 4
                    #             line_index2 += 1

                    # Process rectangles
                    obj_list = []
                    for rect in rects:
                        if len(rect) > 0:
                            res = pyds.NvDsInferObjectDetectionInfo()
                            # Extract bounding box information from stats
                            rect_x1 = rect[0]
                            rect_y1 = rect[1]
                            x2 = rect[2]
                            y2 = rect[3]
                            rect_width = x2 - rect_x1
                            rect_height = y2 - rect_y1
                            res.left = int(rect_x1)
                            res.top = int(rect_y1)
                            res.width = int(rect_width)
                            res.height = int(rect_height)
                            obj_list.append(res)

                    # for circle in circles:
                    #     if len(circle) > 0 and circle_index < 16: # deepstream only supports 18 circles
                    #         circle_params = display_meta.circle_params
                    #         circle_params[circle_index].xc,circle_params[circle_index].yc = circle[0]  # X-coordinate of the circle center
                    #         circle_params[circle_index].radius = circle[1]  # Radius of the circle
                    #         circle_params[circle_index].circle_color.set(1.0, 1.0, 1.0, 0.5)  # Blue color with alpha
                    #         circle_params[circle_index].bg_color.set(1.0, 1.0, 1.0, 0.2)
                    #         circle_params[circle_index].has_bg_color = True
                    #         circle_index += 1
                    # display_meta.num_circles = circle_index  # Set the number of circles

                    for obj in obj_list:
                        add_overlay_meta_to_frame(obj, batch_meta, frame_meta, label_names)
                    
                    display_meta.num_lines = line_index
                    display_meta.num_rects = rect_index
                    display_meta2.num_lines = line_index2

                    """#===========line, rect, arrow and text==========
                    line_params = display_meta.line_params
                    rect_params = display_meta.rect_params

                    # Draw a line
                    line_params[0].x1 = 50
                    line_params[0].y1 = 50
                    line_params[0].x2 = 300
                    line_params[0].y2 = 300
                    line_params[0].line_color.set(0.0, 1.0, 0.0, 0.8)  # Green color with alpha
                    line_params[0].line_width = 4
                    display_meta.num_lines = 1

                    # Draw a rectangle
                    rect_params[0].left = 100
                    rect_params[0].top = 200
                    rect_params[0].width = 150
                    rect_params[0].height = 100
                    rect_params[0].border_width = 5
                    rect_params[0].border_color.set(1.0, 0.0, 0.0, 0.5)  # Red color
                    display_meta.num_rects = 1


                    

                    # Draw a circle
                    circle_params = display_meta.circle_params
                    circle_params[0].xc = 400  # X-coordinate of the circle center
                    circle_params[0].yc = 300  # Y-coordinate of the circle center
                    circle_params[0].radius = 50  # Radius of the circle
                    circle_params[0].circle_color.set(0.0, 0.0, 1.0, 0.5)  # Blue color with alpha
                    display_meta.num_circles = 1  # Set the number of circles to 1        



                    # Draw an arrow
                    arrow_params = display_meta.arrow_params
                    arrow_params[0].x1 = source_id * 250  # Starting X-coordinate
                    arrow_params[0].y1 = 200  # Starting Y-coordinate
                    arrow_params[0].x2 = 300  # Ending X-coordinate
                    arrow_params[0].y2 = 400  # Ending Y-coordinate
                    arrow_params[0].arrow_width = 5  # Width of the arrow shaft
                    arrow_params[0].arrow_color.set(1.0, 0.5, 0.0, 0.8)  # Orange color with alpha
                    arrow_params[0].arrow_head = pyds.NvOSD_Arrow_Head_Direction.END_HEAD  # Arrowhead at the end
                    display_meta.num_arrows = 1  # Set the number of arrows to 1"""

                    display_meta.num_labels = 1  # Number of text elements
                    # Draw text
                    text_params = display_meta.text_params[0]
                    text_params.display_text = "(*A*I*)"  # Text to display
                    # FILE_LOGGER.info(pyds.get_string(text_params.display_text))

                    # Set text position
                    text_params.x_offset = 10  # X-coordinate for the text
                    text_params.y_offset = 10 # Y-coordinate for the text

                    # Set text font properties
                    text_params.font_params.font_name = "Serif"  # Font type
                    text_params.font_params.font_size = 10  # Font size
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



def delete_existing_bin_from_pipeline(uridecodebin,source_id):
    global streammux,pipeline,cam_idx_to_source_bin_map,lock,condition,to_recover_set
    with lock:
        FILE_LOGGER.info("waiting to get condition")
        # condition.wait()
        to_recover_set.remove(source_id)
        condition.notify_all()
    state_return = uridecodebin.set_state(Gst.State.NULL)

    if state_return == Gst.StateChangeReturn.SUCCESS:
        pad_name = "sink_%u" % source_id
        #Retrieve sink pad to be released
        sinkpad =streammux.get_static_pad(pad_name)
        if sinkpad:
            #Send flush stop event to the sink pad, then release from the streammux
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            streammux.release_request_pad(sinkpad)
        #Remove the source bin from the pipeline
        pipeline.remove(uridecodebin)

    elif state_return == Gst.StateChangeReturn.FAILURE:
        FILE_LOGGER.info("STATE CHANGE FAILURE\n")
    
    elif state_return == Gst.StateChangeReturn.ASYNC:
        state_return = uridecodebin.get_state(Gst.CLOCK_TIME_NONE)
        pad_name = "sink_%u" % source_id
        sinkpad =streammux.get_static_pad(pad_name)
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        FILE_LOGGER.info("STATE CHANGE ASYNC\n")
        pipeline.remove(uridecodebin)
    FILE_LOGGER.info(f"uridecodebin with source_id:{source_id} is removed successfully")



def recovery_loop(decoder_configs):
    global to_recover_set,pipeline
    while(True):
        time.sleep(5)
        FILE_LOGGER.info(f"list of cameras that needs to be recovered:{list(to_recover_set)}")
        try:
            for index in to_recover_set.copy():
                FILE_LOGGER.info(f"---------------%Recovering cam-id:{index}%----------------")
                FILE_LOGGER.info(f"Recovery is running, for cam-id:{index}")
                # time.sleep(5)
                FILE_LOGGER.info(f"deleting the current decoder, for cam-id:{index}")
                delete_existing_bin_from_pipeline(cam_idx_to_source_bin_map[index],index)
                time.sleep(0.5)
                FILE_LOGGER.info(f"adding the new decoder, for cam-id:{index}")
                bin = create_source_bin(index,decoder_configs)
                pipeline.add(bin)
                cam_idx_to_source_bin_map[index] = bin
                time.sleep(0.5)
                bin.set_state(Gst.State.PLAYING)
                FILE_LOGGER.info(f"set new bin to PLAYING state for cam-id:{index}")
                time.sleep(0.5)
                status, state, pending = pipeline.get_state(0)
                FILE_LOGGER.info(f"Pipeline state after adding new bin for cam-id:{index}: status={status}, state={state}, pending={pending}")
                FILE_LOGGER.info(f"playing the new decoder, for cam-id:{index}")
                pipeline.set_state(Gst.State.PLAYING)
                time.sleep(3)
        except:
            traceback.print_exc()
            FILE_LOGGER.error("Error occured in the recovery loop",exc_info=True)
                




def bus_call( bus, message, loop):
    """
    This is call back from gstreamer
    
    This method takes 1 parameter and access the messages that is posted on pipeline's bus.
    
    :param loop: this is added from mainloop so that u can kill the loop to kill entire pipeline from messges
    :return: None
    """
    global to_recover_set,lock,condition,status_queue
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        FILE_LOGGER.info("I am here at the EOS part")
        # loop.quit()
        FILE_LOGGER.info(f"The EOS recieved from the triton Pipeline, Exiting the application")
        status_json = {
                        "title" : "Video stopped",
                        "type" : "info",
                        "message" : "Video pipeline exited, The EOS recieved from the triton Pipeline, Exiting the application",
                        "time" : time.ctime(),
                        "source" : "Video pipeline"
                    }
        status_queue.put(status_json)
        # self.stop_ai_loop_event.set()
        # restart_the_aivs_container(config.KILL_FILE)

    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        FILE_LOGGER.info(f"Warning: {err}: {debug}")
    elif t == Gst.MessageType.ERROR:

        err, debug = message.parse_error()
        FILE_LOGGER.info(f"Error: {err}: {debug}")
        src_element = message.src
        parent_obj = src_element.get_parent()
        if parent_obj:
            try:
                parent_obj_name = parent_obj.name
                FILE_LOGGER.info(f"parent object name is {parent_obj_name}")
                if parent_obj.name.startswith("pipeline") or parent_obj.name.startswith("decodebin"):
                    if parent_obj.name.startswith("decodebin"):
                        print("decodebin error")
                    FILE_LOGGER.info("Error! exiting the app parent is pipeline")
                    status_json = {
                        "title" : "Video stopped",
                        "type" : "critical",
                        "message" : f"Video pipeline exited, This is the debugging information:{parent_obj_name}:-[pipeline]--> Error: {err}: {debug} ",
                        "time" : time.ctime(),
                        "source" : "Video pipeline"
                    }
                    # self.status_queue.put(status_json)
                    # self.stop_ai_loop_event.set()
                    # restart_the_aivs_container(config.KILL_FILE)
                    sys.exit(1)
                else:
                    splitted_array = parent_obj.name.split("-")
                    if len(splitted_array) == 2:
                        _,stream_id = splitted_array
                        FILE_LOGGER.info(f"the stream droped for :{stream_id}, setting the recovery event for the uri decoder")
                        stream_id = int(stream_id)
                        with lock:
                            # condition.wait()
                            to_recover_set.add(stream_id)
                            FILE_LOGGER.info("added")
                            condition.notify_all()
                    else:
                        FILE_LOGGER.info(f"Some Error {parent_obj.name}")
                        status_json = {
                            "title" : "Plese check",
                            "type" : "info",
                            "message" : f"Video pipeline exited, This is the debugging information:{parent_obj_name}:-[pipeline]--> Error: {err}: {debug} ",
                            "time" : time.ctime(),
                            "source" : "Video pipeline"
                        }
                        status_queue.put(status_json)
                        # self.stop_ai_loop_event.set()
                        # restart_the_aivs_container(config.KILL_FILE)
                        # sys.exit(1)
            except Exception as e:
                FILE_LOGGER.info("Error! exiting the app ",exc_info=True)
                status_json = {
                        "title" : "Video stopped",
                        "type" : "critical",
                        "message" : f"Video pipeline exited, This is the error: {traceback.format_exc()}, error from element:{parent_obj_name}:-[pipeline]--> Error: {err}: {debug} ",
                        "time" : time.ctime(),
                        "source" : "Video pipeline"
                    }
                # self.status_queue.put(status_json)
                # self.stop_ai_loop_event.set()
                # restart_the_aivs_container(config.KILL_FILE)
                sys.exit(1)
    

    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        #Check for stream-eos message
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                #Set eos status of stream to True, to be deleted in delete-sources
                FILE_LOGGER.info("Got EOS from stream %d" % stream_id)
                with lock:
                    # condition.wait()
                    to_recover_set.add(stream_id)
                    FILE_LOGGER.info(f"added {stream_id} into recovery_Set")
                    condition.notify_all()

    return True


#===========================additions===========================

def cb_newpad(decodebin, decoder_src_pad, data, source_id):
    global streammux
    FILE_LOGGER.info(f"In cb_newpad for source_id: {source_id}")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the nvvideoconvert element
            nvvideoconvert = source_bin.get_by_name("nvvideoconvert")
            if not nvvideoconvert:
                sys.stderr.write("Unable to get nvvideoconvert element from source bin\n")
            
            # Link the decodebin pad to nvvideoconvert's sink pad
            if not decoder_src_pad.link(nvvideoconvert.get_static_pad("sink")) == Gst.PadLinkReturn.OK:
                sys.stderr.write("Failed to link decoder src pad to nvvideoconvert sink pad\n")
            
            # Get the queue element
            queue = source_bin.get_by_name("queue")
            if not queue:
                sys.stderr.write("Unable to get queue element from source bin\n")
            
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(queue.get_static_pad("src")):
                sys.stderr.write("Failed to link queue src pad to source bin ghost pad\n")
            pad_name = "sink_%u" % source_id
            #Get a sink pad from the streammux, link to decodebin
            sinkpad = streammux.get_request_pad(pad_name)
            if not sinkpad:
                sinkpad = streammux.get_static_pad(pad_name)    

            if bin_ghost_pad.link(sinkpad) == Gst.PadLinkReturn.OK:
                FILE_LOGGER.info("Decodebin linked to pipeline")                          
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")



def decodebin_child_added(child_proxy,Object,name,user_data,decoder_configs):
    # print(f"{user_data} in decodebin_child_added")
    FILE_LOGGER.info(f"Decodebin child added:{name}")
    if(name.find("source") != -1):
        # Object.connect("child-added",sourcebin_child_added,user_data)
        if 'GstFileSrc' not in str(Object):
            Object.set_property("latency", 400)
            Object.set_property("drop-on-latency", "true")
            Object.set_property("protocols", "tcp")
    
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data,decoder_configs)
    if(name.find("nvv4l2decoder") != -1):
        Object.set_property("drop-frame-interval", 0)
        Object.set_property("discard-corrupted-frames", True)
        FILE_LOGGER.info("Setting discard-corrupted-frames to True")


def create_source_bin(index,decoder_configs):
    # print("Creating source bin")
    
    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source_bin-%02d" % index
    FILE_LOGGER.info(f"Creating source bin with name: {bin_name}")
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri_decode_bin-{index}")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    filename = cam_idx_to_uri_map[index]
    # We set the input uri to the source element
    if "rtsp" in filename:
        uri_decode_bin.set_property("uri", filename)
    else:
        if not filename.startswith("file://"):
            filename = f"file:///{config.MP4_LOCATION}/{filename}"
        FILE_LOGGER.info(f"new file name for DE-({index}) is {filename}")
        uri_decode_bin.set_property("uri", filename)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin,index)
    uri_decode_bin.connect("child-added", decodebin_child_added,index,decoder_configs)

    # Create elements for nvvideoconvert, videorate, and queue
    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert")
    videorate = Gst.ElementFactory.make("videorate", "videorate")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    nvvideoconvert2 = Gst.ElementFactory.make("nvvideoconvert", "nvvideoconvert2")
    capsfilter2 = Gst.ElementFactory.make("capsfilter", "capsfilter2")
    queue = Gst.ElementFactory.make("queue", "queue")
    # queue_before_nvconv = Gst.ElementFactory.make("queue", "queue_before_nvconv")

    if not nvvideoconvert or not videorate or not capsfilter or not nvvideoconvert2 or not capsfilter2 or not queue:
        sys.stderr.write(" Unable to create elements for source bin \n")

    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),framerate=25/1")
    capsfilter.set_property("caps", caps)

    caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    capsfilter2.set_property("caps", caps2)

    videorate.set_property("max-rate", 25)
    videorate.set_property("drop-only", False)
    videorate.set_property("max-duplication-time", 20000000)

    # Add elements to the bin
    nbin.add(uri_decode_bin)
    # nbin.add(queue_before_nvconv)
    nbin.add(nvvideoconvert)
    nbin.add(videorate) 
    nbin.add(nvvideoconvert2)
    nbin.add(capsfilter)
    nbin.add(capsfilter2)
    nbin.add(queue)

    # Link elements
    uri_decode_bin.link(nvvideoconvert)
    nvvideoconvert.link(videorate)
    videorate.link(nvvideoconvert2)
    nvvideoconvert2.link(capsfilter2)
    capsfilter2.link(capsfilter)
    capsfilter.link(queue)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

#=========================additions end=============================


# def create_source_bin(index, filename,decoder_configs):
#     global streammux
#     bin_name="source_bin-%02d" % index
#     # Source element for reading from the uri.
#     # We will use decodebin and let it figure out the container format of the
#     # stream and the codec and plug the appropriate demux and decode plugins.
#     bin=Gst.ElementFactory.make("uridecodebin", bin_name)
#     bin.set_property('force-sw-decoders','false')
#     if not bin:
#         sys.stderr.write(" Unable to create uri decode bin \n")
#     # We set the input uri to the source element
    
#     if "rtsp" in filename:
#         bin.set_property("uri", filename)
#     else:
#         if not filename.startswith("file://"):
#             filename = f"file:///{config.MP4_LOCATION}/{filename}"
#         FILE_LOGGER.info(f"new file name for DE-({index}) is {filename}")
#         bin.set_property("uri", filename)
#     # Connect to the "pad-added" signal of the decodebin which generates a
#     # callback once a new pad for raw data has been created by the decodebin
#     bin.connect("pad-added",cb_newpad,index)
#     bin.connect("child-added",decodebin_child_added,index,decoder_configs)
#     #Set status of the source to enabled        
#     return bin    

# def cb_newpad(decodebin,pad,data):
#     global streammux
#     FILE_LOGGER.info("In cb_newpad\n")
#     FILE_LOGGER.info(f"in the new pad added callbak for cam-id:{data}")
#     caps=pad.get_current_caps()
#     gststruct=caps.get_structure(0)
#     gstname=gststruct.get_name()

#     # Need to check if the pad created by the decodebin is for video and not
#     # audio.
#     FILE_LOGGER.info(f"gstname={gstname}")
#     if(gstname.find("video")!=-1):
#         source_id = data
#         pad_name = "sink_%u" % source_id
#         #Get a sink pad from the streammux, link to decodebin
#         sinkpad = streammux.get_request_pad(pad_name)
#         if not sinkpad:
#             sinkpad = streammux.get_static_pad(pad_name)

#         if pad.link(sinkpad) == Gst.PadLinkReturn.OK:
#             FILE_LOGGER.info("Decodebin linked to pipeline")
#         else:
#             sys.stderr.write("Failed to link decodebin to pipeline\n")




def main(args):
    global pipeline,streammux,status_queue,FOR_RAMI
    N_Channels = len(args)
    
    global cam_idx_to_python_idx_map,cam_idx_to_source_bin_map,cam_idx_to_uri_map

    List_of_Cameras_indexes = []

    overlay_timeout = 3 # 3 seconds

    batchsize = N_Channels
    resolutions_list = []
    resolutions = (1920, 1080)
    decoder_configs=args

    


    most_common_resolution = (1920, 1080)
    FILE_LOGGER.info(f"Most common resolution: {most_common_resolution}")


    camera_indexes = [str(i+1) for i in range(len(decoder_configs))]
    camera_indexes_name = ",".join(camera_indexes)
    List_of_Cameras_indexes=[int(i+1) for i in range(len(decoder_configs))]
    cam_idx_to_python_idx_map = {idx: i for i, idx in enumerate(List_of_Cameras_indexes)}   
    FILE_LOGGER.info(List_of_Cameras_indexes) 

    width,height = most_common_resolution
    # Create the FOR_RAMI dictionary and initialize with CuPy arrays.
    FOR_RAMI = {
        cam_idx: cp.zeros((height, width, 3,1), dtype=cp.float32) 
        for cam_idx in List_of_Cameras_indexes
    }

    # Print the FOR_RAMI dictionary keys to confirm initialization.
    FILE_LOGGER.info(f"FOR_RAMI initialized for camera indexes:{list(FOR_RAMI.keys())}")
    

    global perf_data
    overlay_buffer=False

    # perf_data = PERF_DATA(len(args))
    number_sources = len(decoder_configs)

    # Standard GStreamer initialization
    Gst.init(None)

    FILE_LOGGER.info("Creating Pipeline \n")
    pipeline = Gst.Pipeline()
    if not pipeline:
        raise RuntimeError("Unable to create Pipeline")

    FILE_LOGGER.info("Creating nvstreammux \n")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        raise RuntimeError("Unable to create NvStreamMux")
    

    # streammux.set_property('sync-inputs', "true")
    # streammux.set_property('align-inputs', "true")
    # streammux.set_property('drop-pipeline-eos', "false")   
    # streammux.set_property('max-latency', 500000000)  

    streammux.set_property('width', width)
    streammux.set_property('height', height)    
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("batch-size", 30)
    streammux.set_property('live-source', 0)
    # streammux.set_property('frame-duration', 0)
    # streammux.set_property('frame-num-reset-on-eos', "true")



    pipeline.add(streammux)

    is_live = False
    for i in range(number_sources):
        FILE_LOGGER.info(f"Creating source_bin {i} \n")
        uri_name = args[i]
        index = i+1
        cam_idx_to_uri_map[index] = uri_name
        if uri_name.startswith("rtsp://"):
            is_live = True

        FILE_LOGGER.info("Creating source bin with index: {} and URI: {}".format(index, uri_name))
        source_bin = create_source_bin(index,decoder_configs)
        if not source_bin:
            raise RuntimeError("Unable to create source bin")

        pipeline.add(source_bin)
        cam_idx_to_source_bin_map[index] = source_bin


    FILE_LOGGER.info("Creating nvstreamdemux \n")
    nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")
    if not nvstreamdemux:
        raise RuntimeError("Unable to create nvstreamdemux")

    pipeline.add(nvstreamdemux)

    # FILE_LOGGER.info("Creating tee \n")
    # tee = Gst.ElementFactory.make("tee", "tee")
    # if not tee:
    #     raise RuntimeError("Unable to create tee")

    # pipeline.add(tee)

    FILE_LOGGER.info("Creating nvvidconv1 \n")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    pipeline.add(nvvidconv1)

    FILE_LOGGER.info("Creating filter_RGBA \n")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter_RGBA")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    pipeline.add(filter1)

    FILE_LOGGER.info("Creating queue_before_demux \n")
    queue_before_demux = Gst.ElementFactory.make("queue", "queue_before_demux")
    if not queue_before_demux:
        raise RuntimeError("Unable to create queue_before_demux")
    pipeline.add(queue_before_demux)


    pgie_src_pad = queue_before_demux.get_static_pad("src")
    # if not pgie_src_pad:
    #     sys.stderr.write(" Unable to get src pad ")
    # else:
    #     pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    # streammux.link(nvvidconv1)
    # nvvidconv1.link(filter1)
    # filter1.link(queue_before_demux)
    # queue_before_demux.link(nvstreamdemux)



  
    watchdog = Gst.ElementFactory.make("watchdog", "watchdog")
    if not watchdog:
        raise RuntimeError("Unable to create watchdog element")
    watchdog.set_property("timeout", 10000) # 10 seconds of idle pipeline will restart the application
    pipeline.add(watchdog)

    streammux.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(watchdog)
    watchdog.link(queue_before_demux)

    pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    pgie.set_property("config-file-path","/opt/nvidia/deepstream/deepstream-7.0/sources/src/dstest1_pgie_inferserver_config_fake_1080_10_expensive.txt")
    pipeline.add(pgie)
    queue_before_demux.link(pgie)

    debug_mode = False

    if debug_mode:
        print("Creating tiler \n ")
        tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
        if not tiler:
            sys.stderr.write(" Unable to create tiler \n")
        tiler_rows = int(math.sqrt(number_sources))
        tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
        tiler.set_property("rows", tiler_rows)
        tiler.set_property("columns", tiler_columns)
        tiler.set_property("width", TILED_OUTPUT_WIDTH)
        tiler.set_property("height", TILED_OUTPUT_HEIGHT)
        pipeline.add(tiler)

        print("Creating nvvidconv \n ")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        if not nvvidconv:
            sys.stderr.write(" Unable to create nvvidconv \n")
        pipeline.add(nvvidconv)

        print("Creating nvosd \n ")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        nvds_sink_pad = nvosd.get_static_pad("sink")
        if not nvds_sink_pad:
            sys.stderr.write(" Unable to get nvdsosd sink pad ")
        else:
            nvds_sink_pad.add_probe(Gst.PadProbeType.BUFFER, nvdsosd_sink_pad_buffer_probe, 0)
        if not nvosd:
            sys.stderr.write(" Unable to create nvosd \n")
        pipeline.add(nvosd)

        platform_info = PlatformInfo()
        if platform_info.is_platform_aarch64():
            print("Creating nv3dsink \n")
            sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        else:
            print("Creating EGLSink \n")
            sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")

        sink.set_property("sync", 1)
        sink.set_property("qos", 0)
        pipeline.add(sink)

        queue_before_demux.link(pgie)
        pgie.link(tiler)
        tiler.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(sink)
    else:
        queue_before_demux.link(pgie)
        pgie.link(nvstreamdemux)
        #Seperate the streams from demux into multiple encoders 
        for i in range(number_sources):
            index = i+1
            FILE_LOGGER.info(f"Creating pipeline for stream {i} \n")

            queue = Gst.ElementFactory.make("queue", f"queue_{i}")
            if not queue:
                raise RuntimeError(f"Unable to create queue for stream {i}")
            pipeline.add(queue)

            queue1 = Gst.ElementFactory.make("queue", f"queue1_{i}")
            queue2 = Gst.ElementFactory.make("queue", f"queue2_{i}")
            queue3 = Gst.ElementFactory.make("queue", f"queue3_{i}")
            queue4 = Gst.ElementFactory.make("queue", f"queue4_{i}")
            # queue5 = Gst.ElementFactory.make("queue", f"queue5_{i}")
            # queue6 = Gst.ElementFactory.make("queue", f"queue6_{i}")

            pipeline.add(queue1,queue2,queue3,queue4)


            nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert_{i}")
            if not nvvideoconvert:
                raise RuntimeError(f"Unable to create nvvideoconvert for stream {i}")
            pipeline.add(nvvideoconvert)

            nvvideoconvert2 = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert2_{i}")
            if not nvvideoconvert2:
                raise RuntimeError(f"Unable to create nvvideoconvert for stream {i}")
            pipeline.add(nvvideoconvert2)    

            nvvideoconvert3 = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert3_{i}")
            if not nvvideoconvert3:
                raise RuntimeError(f"Unable to create nvvideoconvert for stream {i}")
            pipeline.add(nvvideoconvert3)               

            if overlay_buffer:
                # Create and configure elements for the delay queue
                nvvideoconvert_delay = Gst.ElementFactory.make("nvvideoconvert", f"nvvideoconvert_delay_{i}")
                if not nvvideoconvert_delay:
                    raise RuntimeError(f"Unable to create nvvideoconvert_delay for stream {i}")
                pipeline.add(nvvideoconvert_delay)

                videoconvert = Gst.ElementFactory.make("videoconvert", f"videoconvert_{i}")
                if not videoconvert:
                    raise RuntimeError(f"Unable to create videoconvert for stream {i}")
                pipeline.add(videoconvert)

                capsfilter = Gst.ElementFactory.make("capsfilter", f"capsfilter_{i}")
                if not capsfilter:
                    raise RuntimeError(f"Unable to create capsfilter for stream {i}")
                capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGBA"))
                pipeline.add(capsfilter)

                delay_queue = Gst.ElementFactory.make("queue", f"delay_queue_{i}")
                if not delay_queue:
                    raise RuntimeError(f"Unable to create delay_queue for stream {i}")
                delay_queue.set_property("max-size-buffers", 10)
                delay_queue.set_property("max-size-time", 2000000000)
                delay_queue.set_property("max-size-bytes", 440401920)
                delay_queue.set_property("min-threshold-time", 2000000000)
                delay_queue.set_property("min-threshold-buffers", 8)
                pipeline.add(delay_queue)

            # encoder = Gst.ElementFactory.make("nvh264enc", f"encoder_{i}")
            encoder = Gst.ElementFactory.make("nvh264enc", f"encoder_{i}")
            if not encoder:
                raise RuntimeError(f"Unable to create encoder for stream {i}")
            # Set properties for high performance
            encoder.set_property("bitrate", 4000)
            encoder.set_property("max-bitrate", 5500)
            encoder.set_property("rc-mode", 2)  # Use Constant Bitrate (CBR) mode
            encoder.set_property("preset", 0)  # Use performance preset
            # encoder.set_property("gop-size", 150)
            # encoder.set_property("spatial-aq", True)
            # encoder.set_property("temporal-aq", True)
            # encoder.set_property("aq-strength", 8)
            # encoder.set_property("bframes", 1)
            # encoder.set_property("b-adapt", False)
            # encoder.set_property("vbv-buffer-size", 500)
            # encoder.set_property("qp-const", 30)
            # encoder.set_property("qp-min", 25)
            # encoder.set_property("qp-max", 40)
            # encoder.set_property("zerolatency", True)  # Minimize latency
            # encoder.set_property("qos", "true")
            pipeline.add(encoder)

            FILE_LOGGER.info("creating rtspclient sink")
            rtspclientsink = Gst.ElementFactory.make("rtspclientsink", f"rtspclientsink{i}")
            if not rtspclientsink:
                sys.stderr.write(" Unable to create udpsink")
            FILE_LOGGER.info(f"rtsp://127.0.0.1:554/video{i}")
            rtspclientsink.set_property("location",f"rtsp://127.0.0.1:554/video{index}")
            rtspclientsink.set_property("protocols","tcp")
            pipeline.add(rtspclientsink)

            padname = f"src_{index}"
            demuxsrcpad = nvstreamdemux.get_request_pad(padname)
            if not demuxsrcpad:
                raise RuntimeError("Unable to create demux src pad")
            queuesinkpad = queue.get_static_pad("sink")
            if not queuesinkpad:
                raise RuntimeError("Unable to create queue sink pad")
            demuxsrcpad.link(queuesinkpad)

            # creating nvosd
            nvdsosd = make_element("nvdsosd", i)
            pipeline.add(nvdsosd)
            nvdsosd.set_property("process-mode", 0)
            nvdsosd.set_property("display-text", 1)

            # Link the elements
            queue.link(nvvideoconvert)

            if overlay_buffer:
                nvvideoconvert.link(nvvideoconvert_delay)
                nvvideoconvert_delay.link(videoconvert)
                videoconvert.link(capsfilter)
                capsfilter.link(delay_queue)
                delay_queue.link(nvvideoconvert3)
            else:
                nvvideoconvert.link(nvvideoconvert3)

            nvvideoconvert3.link(queue1)

            # videorate = Gst.ElementFactory.make("videorate", f"videorate_{i}")
            # if not videorate:
            #     raise RuntimeError(f"Unable to create videorate for stream {i}")
            # videorate.set_property("max-rate", 25)
            # pipeline.add(videorate)

            # FILE_LOGGER.info("Creating filter_framerate \n")
            # caps2 = Gst.Caps.from_string("video/x-raw,framerate=25/1")
            # filter2 = Gst.ElementFactory.make("capsfilter", f"filter_framerate_{i}")
            # if not filter2:
            #     sys.stderr.write(" Unable to get the caps filter2 \n")
            # filter2.set_property("caps", caps2)
            # pipeline.add(filter2)

            # nvvideoconvert3.link(videorate)
            # videorate.link(filter2)
            # filter2.link(queue1)

            queue1.link(nvdsosd)
            nvdsosd.link(queue2)
            queue2.link(nvvideoconvert2)
            nvvideoconvert2.link(queue3)


            nvds_sink_pad = nvdsosd.get_static_pad("sink")
            if not nvds_sink_pad:
                sys.stderr.write(" Unable to get nvdsosd sink pad ")
            else:
                nvds_sink_pad.add_probe(Gst.PadProbeType.BUFFER, nvdsosd_sink_pad_buffer_probe, i)

            queue3.link(encoder)
            encoder.link(queue4)
            queue4.link(rtspclientsink)

    # Create A diagram of our architecture 



    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    FILE_LOGGER.info("Now playing...")
    FILE_LOGGER.info("Initializing GLib MainLoop")


    pipeline.set_state(Gst.State.PAUSED)
    # Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
    
    time.sleep(0.5)


    FILE_LOGGER.info("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    
    FILE_LOGGER.info("Starting recovery thread")

    recovery_thread = threading.Thread(target=recovery_loop,args=(decoder_configs,),daemon=True)
    recovery_thread.start()  

    FILE_LOGGER.info("Sleeping for 10 seconds before starting AI thread (if applicable)")
    time.sleep(2)
    try:
        Gst.debug_bin_to_dot_file(pipeline, Gst.DebugGraphDetails.ALL, f"pipeline_{time.strftime('%Y_%m_%d__%H_%M_%S')}")
        FILE_LOGGER.info("Pipeline debug graph created successfully.")
    except Exception as e:
        FILE_LOGGER.error("Failed to create pipeline debug graph", exc_info=True)
    analytics_status = True
    FILE_LOGGER.info(f"Analytics status: {analytics_status}")

    ai_killer_event = threading.Event()
    # if analytics_status != "OFF":
    #     FILE_LOGGER.info("Starting AI thread")
    #     ai_thread = threading.Thread(target=M, args=(most_common_resolution, List_of_Cameras_indexes,), daemon=True)
    #     ai_thread.start()
    # else:
    #     FILE_LOGGER.info("AI thread not started as Analytics is OFF or not specified")


    

    main_loop_thread = threading.Thread(target=loop.run,daemon=True)
    main_loop_thread.start()


    # recovery_thread.join()
    # import pdb;pdb.set_trace()
    FILE_LOGGER.info("Heartbeat: monitoring application health")

    while True:
        time.sleep(5)
        FILE_LOGGER.info("heart beat!!")
        state_change_return, state, pending = pipeline.get_state(0)
        if recovery_thread.is_alive():
            FILE_LOGGER.info("Recovery thread is alive and running.")
        else:
            FILE_LOGGER.warning("Recovery thread has stopped.")
        FILE_LOGGER.info(f"Pipeline state: {state}")
        if os.path.exists(config.KILL_FILE):
            FILE_LOGGER.info("Teardown file detected. Pausing pipeline and exiting system.")
            print(f"Bye Bye:{time.ctime()}")
            time.sleep(50)
            ai_killer_event.set()
            break
            # pipeline.set_state(Gst.State.PAUSED)
            # restart_the_aivs_container(config.KILL_FILE)


import argparse
if __name__ == '__main__':
    platform_info = PlatformInfo()
    if platform_info.is_integrated_gpu():
        sys.stderr.write ("\nThis app is not currently supported on integrated GPU. Exiting...\n\n\n\n")
        sys.exit(1)
    #stream_paths = parse_args()

    
    # from Rules_Reader import Master
    # S1=[1080,1920,3]

    # Master2025=Master(list(np.arange(1,11)),S1,'Thermal','Static')
    #time.sleep(5)
    stream_paths =1*["file:///mp4_ingest/1.mp4"] + ["rtsp://192.168.10.135:8555/video1"] # simulating 10 cameras
    sys.exit(main(stream_paths))

