
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
import config




class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """


    def initialize(self, args):
        
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        """
        self.source_id_q = Queue(maxsize=100)
        # CUDA kernel that hogs GPU
        kernel_code = r'''
        extern "C" __global__
        void high_load_kernel(float* out, int iterations) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            float val = 0.0f;

            for (int i = 0; i < iterations; ++i) {
                val += sinf(i * 0.0001f) * cosf(i * 0.00005f);
            }

            out[idx] = val;
        }
        '''

        # Compile kernel once
        module = cp.RawModule(code=kernel_code)
        self.kernel = module.get_function('high_load_kernel')

        threading.Thread(target=self.expensive_thread, daemon=True).start()



    def expensive_thread(self):
        # Simulation parameters
        NUM_THREADS = 1024 * 1024   # 1 million
        ITERATIONS = 10000         # simulate expensive computation
        BLOCK_SIZE = 256
        GRID_SIZE = (NUM_THREADS + BLOCK_SIZE - 1) // BLOCK_SIZE
        TARGET_FPS = 25
        FRAME_INTERVAL = 1.0 / TARGET_FPS

        # Allocate output buffer
        output = cp.zeros(NUM_THREADS, dtype=cp.float32)

        print("Starting GPU overload simulation at 25 FPS...")
        try:
            while True:
                start_time = time.time()

                # Launch synthetic high-load kernel
                self.kernel((GRID_SIZE,), (BLOCK_SIZE,), (output, cp.int32(ITERATIONS)))
                cp.cuda.Device().synchronize()

                # Enforce real-time frame pacing
                elapsed = time.time() - start_time
                sleep_time = max(0, FRAME_INTERVAL - elapsed)
                random_delay = cp.random.uniform(0.02, 0.03).item()
                time.sleep(random_delay)

                print(f"Frame done in {elapsed:.3f} s | sleeping {sleep_time:.3f} s")

        except:
            print("❎ Simulation stopped.")
            traceback.print_exc()





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
            batch = cp.fromDlpack(input_tensor.to_dlpack())
            batch_size = batch.shape[0]

            rects_for_all_channels = []
            batch_size = batch.shape[0]

            # indexes_from_pipeline = self.find_order()
            # print(f"indexes_from_pipeline={indexes_from_pipeline}")
            try:

                cv2.imshow("Tiler Debug Window from Triton", cp.asnumpy(batch[0]).astype(np.uint8))
                key = cv2.waitKey(1)
            except Exception as e:
                print("Exception occurred")
                traceback.print_exc()         


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
