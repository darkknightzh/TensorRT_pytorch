import argparse
import numpy as np
import sys
from PIL import Image

#from model_function import Example_Model
from train import Net
from engine_function import *
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit    # 此句代码中未使用，但是必须有。this is useful, otherwise stream = cuda.Stream() will cause 'explicit_context_dependent failed: invalid device context - no currently active context?'

import calibrator
import collections

def normalize(input, mean, std):
    input = input/255.0
    return (input - mean)/std


def test_pytorch():
    net = Net()
    checkpoint_path = 'mnist_cnn_3.pt'
    net.load_state_dict(torch.load(checkpoint_path))

    correct = 0
    data, targets = torch.load('data/MNIST/processed/test.pt')
    data, targets = data.numpy(), targets.numpy()
    with torch.no_grad():
        for index in range(len(data)):
            if index%100 == 0:
                print(index, len(data))
            img, target = data[index], int(targets[index])
            img = normalize(img, mean=(0.1307,), std=(0.3081,))
            img = img[np.newaxis, np.newaxis, :,:]
            img = torch.from_numpy(img)
            output = net(img.float())
            pred = np.argmax(output.numpy(), axis=1)
            correct += pred[0] == target

    print('\nAccuracy: {}  ({:.0f}%)\n'.format(correct, 100. * correct / len(targets)))


def export_onnx():
    net = Net()
    checkpoint_path = 'mnist_cnn_3.pt'
    net.load_state_dict(torch.load(checkpoint_path))
    saveONNX(net, 'mnist_cnn_3.onnx', 1, 28, 28)


def test_onnx_fp32():
    #torch.load('check_point.pth')   # 如果结果不对，加上这句话
    with build_engine('mnist_cnn_3.onnx') as engine, engine.create_execution_context() as context:
        #save_engine(engine, "mnist_cnn_3_fp32.engine")
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        correct = 0
        data, targets = torch.load('data/MNIST/processed/test.pt')
        data, targets = data.numpy(), targets.numpy()
        for index in range(len(data)):
            if index % 100 == 0:
                print(index, len(data))
            img, target = data[index], int(targets[index])
            img = normalize(img, mean=(0.1307,), std=(0.3081,))

            img_numpy = img.ravel().astype(np.float32)
            np.copyto(inputs[0].host, img_numpy)
            output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            output = [np.reshape(stage_i, (10)) for stage_i in output]  # 有多个输出时遍历

            pred = np.argmax(output, axis=1)
            correct += pred[0] == target

        print('\nAccuracy: {}  ({:.0f}%)\n'.format(correct, 100. * correct / len(targets)))

def test_onnx_fp32_engine():
    #torch.load('check_point.pth')   # 如果结果不对，加上这句话
    with load_engine("mnist_cnn_3_fp32.engine") as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        correct = 0
        data, targets = torch.load('data/MNIST/processed/test.pt')
        data, targets = data.numpy(), targets.numpy()
        for index in range(len(data)):
            if index % 100 == 0:
                print(index, len(data))
            img, target = data[index], int(targets[index])
            img = normalize(img, mean=(0.1307,), std=(0.3081,))

            img_numpy = img.ravel().astype(np.float32)
            np.copyto(inputs[0].host, img_numpy)
            output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            output = [np.reshape(stage_i, (10)) for stage_i in output]  # 有多个输出时遍历

            pred = np.argmax(output, axis=1)
            correct += pred[0] == target

        print('\nAccuracy: {}  ({:.0f}%)\n'.format(correct, 100. * correct / len(targets)))


def test_onnx_int8():
    #torch.load('mnist_cnn_3.pth')    # 如果结果不对，加上这句话
    calibration_cache = "mnist_cnn_3.cache"
    calib = calibrator.ExampleEntropyCalibrator(datafolder='data/MNIST/processed/test.pt', cache_file=calibration_cache, c=1, h=28, w=28)

    with build_engine_int8('mnist_cnn_3.onnx', calib) as engine, engine.create_execution_context() as context:
        #save_engine(engine, "mnist_cnn_3_int8.engine")
        inputs, outputs, bindings, stream = allocate_buffers(engine)  # 最好在下面do_inference之前分配一次内存即可

        correct = 0
        data, targets = torch.load('data/MNIST/processed/test.pt')
        data, targets = data.numpy(), targets.numpy()
        for index in range(len(data)):
            if index % 100 == 0:
                print(index, len(data))
            img, target = data[index], int(targets[index])
            img = normalize(img, mean=(0.1307,), std=(0.3081,))

            img_numpy = img.ravel().astype(np.float32)
            np.copyto(inputs[0].host, img_numpy)
            output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            output = [np.reshape(stage_i, (10)) for stage_i in output]  # 有多个输出时遍历

            pred = np.argmax(output, axis=1)
            correct += pred[0] == target

        print('\nAccuracy: {}  ({:.0f}%)\n'.format(correct, 100. * correct / len(targets)))
        

def test_onnx_int8_engine():
    #torch.load('mnist_cnn_3.pth')
    with load_engine("mnist_cnn_3_int8.engine") as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)   # 最好在下面do_inference之前分配一次内存即可

        correct = 0
        data, targets = torch.load('data/MNIST/processed/test.pt')
        data, targets = data.numpy(), targets.numpy()
        for index in range(len(data)):
            if index % 100 == 0:
                print(index, len(data))
            img, target = data[index], int(targets[index])
            img = normalize(img, mean=(0.1307,), std=(0.3081,))

            img_numpy = img.ravel().astype(np.float32)
            np.copyto(inputs[0].host, img_numpy)
            output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            output = [np.reshape(stage_i, (10)) for stage_i in output]  # 有多个输出时遍历

            pred = np.argmax(output, axis=1)
            correct += pred[0] == target

        print('\nAccuracy: {}  ({:.0f}%)\n'.format(correct, 100. * correct / len(targets)))


def main():
    test_pytorch()
    #export_onnx()
    #test_onnx_fp32()
    #test_onnx_fp32_engine()
    #test_onnx_int8()
    #test_onnx_int8_engine()


if __name__ == '__main__':
    main()

