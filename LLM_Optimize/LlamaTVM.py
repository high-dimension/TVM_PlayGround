from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import tvm
from tvm import relax
from tvm.runtime import ShapeTuple

from LLM_Optimize.LlamaConfig import LlamaConfig
from LLM_Optimize.LlamaForCasualLM import LlamaForCasualLM
from LLM_Optimize.context import device, target


class LlamaTVM:
    def __init__(self, weight_path: Path, _device):
        self.__model_config = LlamaConfig()
        self.__model = LlamaForCasualLM(self.__model_config)
        self.__model.to("float16")

        self.__mod, self.__named_params = self.__model.export_tvm(spec=self.__model.get_default_spec())
        self.__params = self.__prepare_weights(weight_path, _device=_device)

        with target:
            self.__ex = relax.build(self.__mod, target, pipeline=relax.get_pipeline("opt_llm"))
            self.__vm = relax.VirtualMachine(self.__ex, _device)

        self.__kv_cache = self.__vm["create_tir_paged_kv_cache"](ShapeTuple([1]),  # max_batch_size=1
                                                                 ShapeTuple([2048]),  # max_total_seq_len=2048
                                                                 ShapeTuple([2048]),  # prefill_chunk_size=2048
                                                                 ShapeTuple([16]),  # page_size=16
                                                                 )

        # mod, named_params = model.export_tvm(spec=model.get_default_spec())
        # prefill_str = mod["prefill"].script()
        # print(*prefill_str.split("\n")[3:20], sep="\n")  # Only show the first 10 lines for demonstration
        # print("        ...")
        #
        # print("\nParameters:")
        # pprint(named_params[:5])

    def __prepare_weights(self, weight_path: Path, _device):
        if not weight_path.exists():
            raise ValueError(f"Weight path: {weight_path} not exist.")

        param_dict = safetensors.torch.load_file(weight_path / "model.safetensors", device="cpu")
        # print(param_dict)

        param_dict = {
            k: v.half().numpy() if v.dtype == torch.bfloat16 else v.numpy()
            for k, v in param_dict.items()
        }
        # print(param_dict.keys())
        self.__named_params = dict(self.__named_params)
        # print(self.__named_params)

        for i in range(self.__model_config.num_hidden_layers):
            # Add QKV in self attention
            attn = f"model.layers.{i}.self_attn"
            # print(f"{attn}.qkv_proj.weight")
            param_dict[f"{attn}.qkv_proj.weight"] = np.concatenate(
                [
                    param_dict.pop(f"{attn}.q_proj.weight"),  # Pop the old parameters to save memory
                    param_dict.pop(f"{attn}.k_proj.weight"),
                    param_dict.pop(f"{attn}.v_proj.weight"),
                ],
                axis=0,
            )
            # Add gates in MLP
            mlp = f"model.layers.{i}.mlp"
            param_dict[f"{mlp}.gate_up_proj.weight"] = np.concatenate(
                [
                    param_dict.pop(f"{mlp}.gate_proj.weight"),
                    param_dict.pop(f"{mlp}.up_proj.weight"),
                ],
                axis=0,
            )
        params = [
            tvm.nd.array(param_dict[k].astype("float16"), device=_device) for k in self.__named_params.keys()
        ]
        return params


def main():
    llama = LlamaTVM(weight_path=Path('/home/hqin/data/TinyLlama/TinyLlamaTinyLlama1.1B-Chat-v1.0'), _device=device)
    print()


if __name__ == '__main__':
    main()
