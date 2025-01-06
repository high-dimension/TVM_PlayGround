import numpy as np
import tvm
from tvm import relax
from tvm.relax.frontend import nn


class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


mod, param_spec = MLPModel().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod.show()

mod = relax.get_pipeline("zero")(mod)

mod.show()

device = tvm.cpu()
target = tvm.target.Target.from_device(device)

ex = relax.build(mod, target)
vm = relax.VirtualMachine(ex, device)

data = np.random.rand(1, 784).astype(np.float32)
tvm_data = tvm.nd.array(data, device=device)

params = [np.random.rand(*param.shape).astype('float32') for _, param in param_spec]
params = [tvm.nd.array(param, device=device) for param in params]
print(vm['forward'](tvm_data, *params))
