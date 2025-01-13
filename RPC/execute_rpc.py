import numpy as np
import tvm
from tvm import rpc, te

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
s = te.create_schedule(B.op)

local_demo = True

if local_demo:
    remote = rpc.LocalSession()
else:
    # The following is my environment, change this to the IP address of your target device
    host = "10.77.1.162"
    port = 9090
    remote = rpc.connect(host, port)

remote.upload('./lib.tar')
func = remote.load_module("lib.tar")

# create arrays on the remote device
dev = remote.cpu()
a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), dev)
b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
# the function will run on the remote device
func(a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)

print('aaaa')
print(a)
print(b)
