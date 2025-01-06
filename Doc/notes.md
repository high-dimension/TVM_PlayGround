## pipeline('zero') 
包含如下4个变化

### LegalizeOps
把Relax中的高级算子，降级为TIR（Tensor Intermedia Representation）中的低级算子

### AnnotateTIROpPattern
注释TIR（Tensor Intermedia Representation） 函数的工作模式，比如kElemWise(逐元素操作)， kInjective(注入式操作)等，不同的函数工作模式，决定了不同优化策略
`注释会加入到函数的属性里 T.func_attr({"op_pattern": 2, ...})`

### FoldConstant
常量折叠

### FuseOps and FuseTIR
根据AnnotateTIROpPattern的输出，先对Relax层的算子做融合，然后把Relax中的基本操作融合到TIR里

---

### relax.transform.MetaScheduleTuneTIR
增对TIR层进行优化，优化方法是建立所有可能的TIR优化搜索空间，找到最优实现

### relax.transform.MetaScheduleApplyDatabase
把 MetaScheduleTuneTIR 找到的最佳调度策略运用到模型上
