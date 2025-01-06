from typing import Tuple

import torch
from torch import nn
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

import IRModule
from IRModule.IGetTvmModel import IGetTvmModel


class TorchModel(nn.Module, IGetTvmModel):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def get(self) -> Tuple[IRModule, dict]:
        example_args = (torch.randn(1, 784, dtype=torch.float32),)

        # Convert the model to IRModule
        with torch.no_grad():
            exported_program = export(TorchModel().eval(), example_args)
            mod_from_torch = from_exported_program(
                exported_program, keep_params_as_input=True, unwrap_unit_return_tuple=True
            )
        mod_from_torch, params_from_torch = relax.frontend.detach_params(mod_from_torch)
        return mod_from_torch, params_from_torch


def main():
    torch_model = TorchModel()
    mod, params = torch_model.get()
    mod.show()

    print(len(params['main']))



if __name__ == "__main__":
    main()
