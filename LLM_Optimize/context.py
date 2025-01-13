import tvm

device = tvm.device("cuda")
target = tvm.target.Target.from_device(device)