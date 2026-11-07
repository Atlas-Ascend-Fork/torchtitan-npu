# FSDP AllGather 缓冲区的计算流分配

该功能将 FSDP 的 AllGather（AG，收集完整参数）暂存输出放到计算流申请，使这些缓冲区能够复用计算流的缓存内存。通信仍在原生通信流执行；开关不改变通信内容和梯度计算。

## 启用方式

开关默认关闭。需要在所有训练进程启动、首次导入 `torchtitan_npu` 之前设置。仅值 `1` 表示开启；运行中修改环境变量不会重新配置已加载的模块。

已验证的 AG 配置：

```bash
export TORCHTITAN_NPU_FSDP_AG_ALLOC_ON_COMPUTE_STREAM=1
export MULTI_STREAM_MEMORY_REUSE=2
# 随后执行原训练命令。
```

ReduceScatter（RS，归约并分片梯度）的分配和跨流保护沿用原有路径。

`MULTI_STREAM_MEMORY_REUSE` 控制 torch_npu 后端的跨流内存保护方式，与 `MULTI_STREAM_MEMORY_RESERVE` 是不同的配置。本功能不会自动覆盖这些后端环境变量。

## 回收行为与限制

- AG 输出沿用原生 FSDP 结果对象的持有周期。正常消费不额外记录 copy-in/AG 流；其他流消费时，最后读取的事件交还申请流。未消费的预取在原生清理入口等待完成，异常路径保留保守的流记录。
- 在已测 torch_npu 版本中，`MULTI_STREAM_MEMORY_REUSE=1` 或 `3` 的同步 AG 旁流路径仍可能因 HCCL 的流记录而延迟回收。这时计算流申请已经生效，并不表示整个功能失效或内存永久泄漏。推荐上述显式 `2` 配置；未设置时的默认行为不能跨版本推断。
- AG 的 dtype 转换及预量化临时张量仍遵循原生路径。
- 自定义通信 allocator、对称内存、CPU offload、GraphTrainer/capture 等组合未完成验证，不能从普通 FSDP 的结果推断其支持情况。

Reserved 表示分配器保留的内存池，Active 表示活跃块统计；二者均不是整卡设备占用。收益取决于负载、内存复用和执行时序，不承诺所有模型都有同等收益。

详细流、分配与释放分析使用 profiler 和带历史记录的 memory snapshot；仅凭一次申请前后 Reserved 差值，不能判断复用收益。
