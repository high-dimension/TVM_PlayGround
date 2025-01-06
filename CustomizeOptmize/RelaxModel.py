import os
import tempfile
from typing import Tuple

import numpy as np
from tvm import dlight as dl

import tvm
from tvm import IRModule, relax
from tvm.relax.frontend import nn


class RelaxModel(nn.Module):
    def __init__(self):
        super(RelaxModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def export_model() -> Tuple[IRModule, dict]:
        input_shape = (1, 784)
        mod, params = RelaxModel().export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})
        return mod, params


def main():
    os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin/"

    mod, params = RelaxModel.export_model()
    print('====Before optimization:')
    mod.show()
    print(params, type(params))

    device = tvm.cuda(0)
    target = tvm.target.Target.from_device(device)

    trials = 200
    with target, tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir)
        mod = tvm.ir.transform.Sequential(
            [
                relax.get_pipeline('zero'),
                dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback()
                ),
                relax.transform.MetaScheduleTuneTIR(work_dir=tmpdir, max_trials_global=trials),
                relax.transform.MetaScheduleApplyDatabase(work_dir=tmpdir),
            ]
        )(mod)
    print('####After optimization')
    mod.show()

    input_shape = (1, 784)
    ex = relax.build(mod, target)

    vm = relax.VirtualMachine(ex, device)
    # Need to allocate data and params on GPU device
    data = tvm.nd.array(np.random.rand(*input_shape).astype("float32"), device)
    gpu_params = [tvm.nd.array(np.random.rand(*p.shape).astype(p.dtype), device) for _, p in params]
    gpu_out = vm["forward"](data, *gpu_params).numpy()
    print(gpu_out)

if __name__ == "__main__":
    main()
