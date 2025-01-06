import numpy as np
import os
import tvm
from tvm import relax
from tvm import dlight as dl

from IRModule.TorchModel import TorchModel


def deploy_on_cpu():
    mod, params = TorchModel().get()
    mod = relax.get_pipeline('zero')(mod)

    device = tvm.cpu()
    target = tvm.target.Target.from_device(device)

    executable = relax.build(mod, target)
    vm = relax.VirtualMachine(executable, device)

    np_data = np.random.rand(1, 784).astype('float32')
    tvm_data = tvm.nd.array(np_data, device=device)
    # *params 解包操作，相当于 func(w1, b1, ...)
    out = vm['main'](tvm_data, *params['main'])
    out = out.numpy()
    print(out)


def deploy_on_cuda():
    os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin/"

    mod, params = TorchModel().get()
    mod = relax.get_pipeline('zero')(mod)
    mod.show()
    print('========================')

    # 使用dlight 优化调度策略后，能看到模型的计算绑定到线程这个级别了，dlight 具体的用法，需要在后续继续了解
    with tvm.target.cuda():
        gpu_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Fallback())(mod)
        # print(gpu_mod)
    target = tvm.target.cuda()
    executable = relax.build(gpu_mod, target=target)
    dev = tvm.cuda()
    vm = relax.VirtualMachine(executable, dev)

    np_data = np.random.rand(1, 784).astype('float32')
    gpu_data = tvm.nd.array(np_data, dev)
    gpu_params = [tvm.nd.array(param, dev) for param in params['main']]
    gpu_out = vm['main'](gpu_data, *gpu_params).numpy()
    gpu_mod.show()
    print(gpu_out)


def main():
    deploy_on_cuda()


if __name__ == "__main__":
    main()
