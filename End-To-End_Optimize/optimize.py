import os
import numpy as np
import torch
import tvm
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from tvm import dlight as dl

os.environ["PATH"] = os.environ["PATH"] + ":/usr/local/cuda/bin/"
torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

# Convert the model to IRModule
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod = from_exported_program(exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True)

mod, params = relax.frontend.detach_params(mod)
mod.show()

TOTAL_TRIALS = 1  # Change to 20000 for better performance if needed

device = tvm.cuda(0)
target = tvm.target.Target.from_device(device)
# target = tvm.target.Target('nvidia/geforce-rtx-4060-ti')  # Change to your target device
work_dir = "tuning_logs"


mod = relax.get_pipeline("static_shape_tuning", target=target, total_trials=TOTAL_TRIALS)(mod)

with tvm.target.cuda():
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), dl.gpu.Fallback())(mod)

# Only show the main function
mod.show()

ex = relax.build(mod, target=target)

vm = relax.VirtualMachine(ex, device)
# Need to allocate data and params on GPU device
gpu_data = tvm.nd.array(np.random.rand(1, 3, 224, 224).astype("float32"), device)
gpu_params = [tvm.nd.array(p, device) for p in params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params).numpy()

print(gpu_out.shape)