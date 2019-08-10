import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit    # 此句代码中未使用，但是必须有。this is useful, otherwise stream = cuda.Stream() will cause 'explicit_context_dependent failed: invalid device context - no currently active context?'

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def saveONNX(model, filepath, c, h, w):
    model = model.cuda()
    dummy_input = torch.randn(1, c, h, w, device='cuda')
    torch.onnx.export(model, dummy_input, filepath, verbose=True)


def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)   # INFO
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if builder.platform_has_fast_fp16:
            print('this card support fp16')
        if builder.platform_has_fast_int8:
            print('this card support int8')

        builder.max_workspace_size = 1 << 30
        with open(onnx_file_path, 'rb') as model:
           parser.parse(model.read())
        return builder.build_cuda_engine(network)

# This function builds an engine from a Caffe model.
def build_engine_int8(onnx_file_path, calib):
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = 1  # calib.get_batch_size()
        builder.max_workspace_size = 1 << 30
        builder.int8_mode = True
        builder.int8_calibrator = calib
        with open(onnx_file_path, 'rb') as model:
           parser.parse(model.read())   # , dtype=trt.float32
        return builder.build_cuda_engine(network)

def save_engine(engine, engine_dest_path):
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
