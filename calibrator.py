#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import tensorrt as trt
import os
import json

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np

# For reading size information from batches
import struct
import torch
import cv2
import math

class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batch_data_dir, cache_file):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        # Get a list of all the batch files in the batch folder.
        self.batch_files = [os.path.join(batch_data_dir, f) for f in os.listdir(batch_data_dir)]

        # Find out the shape of a batch and then allocate a device buffer of that size.
        self.shape, _, _ = self.read_batch_file(self.batch_files[0])
        # Each element of the calibration data is a float32.
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)

        # Create a generator that will give us batches. We can use next() to iterate over the result.
        def load_batches():
            for f in self.batch_files:
                shape, data, labels = self.read_batch_file(f)
                yield shape, data, labels
        self.batches = load_batches()

    # This function is used to load calibration data from the calibration batch files.
    # In this implementation, one file corresponds to one batch, but it is also possible to use
    # aggregate data from multiple files, or use only data from portions of a file.
    def read_batch_file(self, filename):
        with open(filename, "rb") as f:
            # Read the first 4 integers. These will be the NCHW dimensions of the data.
            shape = tuple(struct.unpack("<L", f.read(trt.int32.itemsize))[0] for _ in range(4))
            # Next read in all the images, where each element of each image is a float32.
            data = f.read(trt.volume(shape) * trt.float32.itemsize)
            # The remainder of the file consists of labels
            labels = f.read()
        return shape, data, labels

    def get_batch_size(self):
        return self.shape[0]

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Get a single batch.
            _, data, _ = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class ExampleEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, datafolder, cache_file, c, h, w):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        data, targets = torch.load(datafolder)
        self.all_files = data.numpy()

        # Find out the shape of a batch and then allocate a device buffer of that size.
        # self.shape, _, _ = self.read_batch_file(self.batch_files[0])
        self.shape = [1, c, h, w]
        # Each element of the calibration data is a float32.
        self.device_input = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)

        # Create a generator that will give us batches. We can use next() to iterate over the result.
        def load_batches():
            for idx in range(len(self.all_files)):
                data = self.read_batch_file(idx)
                yield data
        self.batches = load_batches()

    # This function is used to load calibration data from the calibration batch files.
    # In this implementation, one file corresponds to one batch, but it is also possible to use
    # aggregate data from multiple files, or use only data from portions of a file.
    def read_batch_file(self, idx):
        def normalize(input, mean, std):
            input = input / 255.0
            return (input - mean) / std

        img = self.all_files[idx]
        normed_img = normalize(img, mean=(0.1307,), std=(0.3081,))
        normed_img = normed_img.astype(np.float32)
        return normed_img

    def get_batch_size(self):
        return self.shape[0]

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        try:
            # Get a single batch.
            data = next(self.batches)
            # Copy to device, then return a list containing pointers to input device buffers.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            print('exception')
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)