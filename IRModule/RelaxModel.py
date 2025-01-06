from typing import Tuple

from tvm.relax.frontend import nn

import IRModule
from IRModule.IGetTvmModel import IGetTvmModel


class RelaxModel(nn.Module, IGetTvmModel):
    def __init__(self):
        super(RelaxModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def get(self) -> Tuple[IRModule, dict]:
        mod_from_relax, params_from_relax = RelaxModel().export_tvm(
            {'forward': {'x': nn.spec.Tensor((1, 784), 'float32')}})
        return mod_from_relax, params_from_relax

def main():
    relax_model = RelaxModel()
    mod, params = relax_model.get()
    mod.show()
    print(mod.get_global_vars())
    print(params)

if __name__ == '__main__':
    main()