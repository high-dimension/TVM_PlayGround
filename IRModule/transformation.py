from tvm import relax

from IRModule.RelaxModel import RelaxModel
from IRModule.TorchModel import TorchModel


def main():
    # orign_mod = RelaxModel()
    orign_mod = TorchModel()

    mod, params_spec = orign_mod.get()
    print('==== Not transform model:')
    mod.show()

    print('==== After LegalizeOps:')
    mod = relax.transform.LegalizeOps()(mod)
    mod.show()

    print('==== After Annotated:')
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod.show()

    print('==== After FoldConst:')
    mod = relax.transform.FoldConstant()(mod)
    mod.show()

    print('==== After Fuse Ops & TIR:')
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod.show()


if __name__ == '__main__':
    main()
