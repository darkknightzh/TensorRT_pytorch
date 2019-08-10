import argparse
import numpy as np
import sys

from model_function import Example_Model
from engine_function import *
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit    # 此句代码中未使用，但是必须有。this is useful, otherwise stream = cuda.Stream() will cause 'explicit_context_dependent failed: invalid device context - no currently active context?'

import calibrator
import collections


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def load_state(net, checkpoint):
    source_state = checkpoint['state_dict']
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

    net.load_state_dict(new_target_state)

def test_pytorch(args):
    net = Example_Model()

    checkpoint_path = 'check_point.pth'
    #torch.save({'state_dict': net.state_dict()}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    load_state(net, checkpoint)

    #input = torch.randn(1, 3, 24, 24)
    #np.save('input.npy', input.cpu())
    input = np.load('input.npy')
    input = torch.tensor(input)

    print(input)

    out = net(input)
    print(out)



def export_onnx(args):
    net = Example_Model()

    checkpoint_path = 'check_point.pth'
    checkpoint = torch.load(checkpoint_path)
    load_state(net, checkpoint)
    saveONNX(net, 'check_point.onnx')


def test_onnx_fp32(args):
    #torch.load('check_point.pth')   # 如果结果不对，加上这句话
    with build_engine('check_point.onnx') as engine, engine.create_execution_context() as context:
        #save_engine(engine, "check_point_fp32.engine")
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        img_numpy = np.load('input.npy')
        img_numpy = img_numpy.ravel().astype(np.float32)
        np.copyto(inputs[0].host, img_numpy)
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        output = [np.reshape(stage_i, (-1, 20, 20)) for stage_i in output]  # 有多个输出时遍历
        print(output)

def test_onnx_fp32_engine(args):
    #torch.load('check_point.pth')   # 如果结果不对，加上这句话
    with load_engine("check_point_fp32.engine") as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        img_numpy = np.load('input.npy')
        img_numpy = img_numpy.ravel().astype(np.float32)
        np.copyto(inputs[0].host, img_numpy)
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        output = [np.reshape(stage_i, (-1, 20, 20)) for stage_i in output]  # 有多个输出时遍历
        print(output)


def test_onnx_int8(args):   # 此处未测试，因为没有数据，正常测试可以
    torch.load('check_point.pth')    # 如果结果不对，加上这句话
    calibration_cache = "check_point.cache"
    calib = calibrator.ExampleEntropyCalibrator(datafolder='your_data_folder',
        file_list='your_json_file.json', cache_file=calibration_cache)

    with build_engine_int8('check_point.onnx', calib) as engine, engine.create_execution_context() as context:
        save_engine(engine, "check_point_int8.engine")
        inputs, outputs, bindings, stream = allocate_buffers(engine)  # 最好在下面do_inference之前分配一次内存即可
        img_numpy = np.load('input.npy')
        img_numpy = img_numpy.ravel().astype(np.float32)
        np.copyto(inputs[0].host, img_numpy)
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        output = [np.reshape(stage_i, (-1, 20, 20)) for stage_i in output]  # 有多个输出时遍历
        print(output)

def test_onnx_int8_engine(args):
    torch.load('check_point.pth')
    with load_engine("check_point_int8.engine") as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)   # 最好在下面do_inference之前分配一次内存即可
        img_numpy = np.load('input.npy')
        img_numpy = img_numpy.ravel().astype(np.float32)
        np.copyto(inputs[0].host, img_numpy)
        output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        output = [np.reshape(stage_i, (-1, 20, 20)) for stage_i in output]  # 有多个输出时遍历
        print(output)


def main(args):
    test_pytorch(args)
    #export_onnx(args)
    #test_onnx_fp32(args)
    #test_onnx_fp32_engine(args)
    #test_onnx_int8(args)
    #test_onnx_int8_engine(args)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

