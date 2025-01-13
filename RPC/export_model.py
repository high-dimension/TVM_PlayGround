import tvm
from tvm import te
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
s = te.create_schedule(B.op)

local_demo = True

if local_demo:
    target = "llvm"
else:
    target = "llvm -mtriple=armv7l-linux-gnueabihf"

func = tvm.build(s, [A, B], target=target, name="add_one")
# save the lib at a local temp folder
temp = utils.tempdir()
# temp = './'
# print(temp)
# path = temp.relpath("lib.tar")
# print(path)
func.export_library('./lib.tar')
