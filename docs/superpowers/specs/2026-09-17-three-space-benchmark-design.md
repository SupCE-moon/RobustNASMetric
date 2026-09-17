# 鲁棒指标三空间基准实验设计

## 一、目标

在用户自己的 `wangjiahao` 容器中建立可复现的 RB40、NB201-64、
DARTS-100 三空间评测环境，补齐 E25 之后指标缺失的实验，并在严格隔离最终测试
排名的前提下，按需设计新指标。

研究目标是让一个有充分依据的指标进入预注册 `All-setting mean` 前五。前五是
努力目标，不是预先保证的结果；全部候选指标和负结果都必须保留并如实报告。

## 二、允许访问的系统与操作边界

- 源容器：`huangyue`。只读取老师提供的项目和
  `/root/robust_metric_lab_nas/shared_benchmarks/robust_metric_v1`。
- 目标容器：`wangjiahao`。
- 目标宿主机目录：`/home/usertest`，它映射为 `wangjiahao` 容器内的
  `/workspace`。
- 已有 `/workspace/DynamicBN20251203` 保持原样，不覆盖、不移动、不删除。
- 不访问其他容器、其他用户目录、其他 GPU 或无关数据集。
- 密码不得写入文件、命令、日志或项目输出。

## 三、目标目录结构

```text
/workspace/robust_metric_lab_nas/
  configs/
  docs/
  scripts/
  src/
  tests/
  outputs/
    student_metrics/wangjiahao/
  shared_benchmarks/
    robust_metric_v1/
```

复制本地项目时排除 `.venv`、`.idea`、Python 缓存、本地模型缓存和数据缓存。
保留历史 CSV 和实验报告，因为它们用于审计 E25 之后的指标覆盖情况，并对比旧协议
与新协议。

## 四、基准快照与材料完整性例外

2026-09-17 检查源材料时，`TRANSFER_STATUS.json.state` 仍为
`IN_PROGRESS`。老师随后明确表示可以直接使用当前材料。本项目接受这一授权，但
不得静默修改状态文件，也不得声称老师的原始材料已经通过验收。

正式评测前必须记录：

- 源容器和源路径；
- 复制时间；
- 原始 `TRANSFER_STATUS.json` 内容及其 SHA-256；
- manifests 文件的 SHA-256；
- 复制后的文件数量与总字节数；
- 验收脚本的标准输出、错误输出和退出码。

在复制后的快照上运行 `verify_student_benchmark_bundle.py`。如果验收通过，则使用
完整三空间协议。如果验收失败，只能继续使用同时满足以下条件的单个空间：manifest
完整、checkpoint 数量正确、校验通过并且实际模型加载测试成功。失败或不完整的空间
必须标记为不可用，不能静默删除后继续计算总平均。

## 五、冻结的评测协议

新基准固定包含八个 setting：

1. RB40 mature；
2. NB201 e0；
3. NB201 e5；
4. NB201 e10；
5. NB201 e25；
6. NB201 e50；
7. DARTS e0；
8. DARTS e100。

排序标签固定使用 manifest 中的最终标签：RB40 使用 AutoAttack robust accuracy，
NB201 使用官方最终 AutoAttack accuracy，DARTS 使用完整测试集 PGD-20 accuracy。

CIFAR-10 测量协议固定为：`sample_offset=0`、`num_samples=64`、
`seed=20260808`、`epsilon=8/255`。同一 setting 内的所有架构必须使用完全相同、
顺序一致的样本。若 e0 指标需要扰动输入，只能使用冻结的 Standard 成熟模型共享
对抗缓存，或预注册的标签无关扰动；禁止让随机初始化的候选模型自行生成对抗样本。

指标方向必须根据数学定义，在连接 robust accuracy 标签之前注册。需要同时报告：

- 原始 Spearman；
- 方向校正后的 Spearman utility；
- Pearson；
- Kendall tau；
- 有效模型数与失败项；
- 运行时间；
- bootstrap 95% 置信区间。

此外，RB40 报告参数量和模型家族控制结果；NB201 报告控制操作数量后的 partial
Spearman。

`All-setting mean` 定义为上述八个 setting 的方向校正 Spearman 简单平均值，并且
只在八项全部有效时计算。缺失 setting 不能记作零，也不能被删除后计算一个虚高的
平均值。

## 六、执行阶段

1. 复制并清点项目代码和 benchmark 快照。
2. 根据项目声明的依赖重建 Linux Python 环境。
3. 运行材料验收脚本，并独立检查文件、checkpoint 数量和校验值。
4. 每个空间选择四个对象做 smoke test，验证模型加载、输入归一化、特征 hook、
   metric 输出和 CSV 格式。
5. 建立 E25 之后“指标 × setting”覆盖矩阵。
6. 使用冻结协议补齐缺失的评测单元。
7. 生成八 setting 基线总表和统计诊断。
8. 只有现有候选没有达到目标时，才基于开发划分设计少量、有明确机制依据的新指标。
9. 在查看最终留出结果之前冻结指标公式、方向、层选择和超参数，然后只进行一次正式
   留出集及完整基准评测。

## 七、新指标的防过拟合规则

新指标可以使用三个空间中确定性的开发划分进行选择。只有在公式、方向、特征层选择
和聚合方式全部冻结后，才允许读取最终留出排序以及官方 `All-setting mean` 表。

禁止以下行为：

- 根据最终结果正负事后翻转指标方向；
- 为提高平均值而删除表现不好的 setting；
- 只保留成功候选、隐瞒失败候选；
- 反复查看最终排行榜并据此调整公式。

## 八、输出规范

每次运行必须创建新的时间戳目录：

```text
outputs/student_metrics/wangjiahao/<metric>/<timestamp>/
```

每次运行至少保存：

- `metric_values.csv`；
- `correlations.csv`；
- `run.json`；
- 完整运行命令；
- 日志和失败行；
- 软件及环境版本；
- 源材料快照元数据；
- Markdown 实验报告。

任何历史结果都不得覆盖。

## 九、完成标准

- 目标目录可以复现，旧项目内容完全未被改动；
- benchmark 快照来源、原始状态和验收结果均已保存；
- 每个声称完成的 setting 均具有完整预期行数和明确的失败审计；
- E25 之后的覆盖缺口已经补齐，或留下可复现的阻塞原因；
- 最终跨空间总表严格使用冻结的八 setting 定义；
- 任何“进入前五”的结论都同时提供不确定性、混杂诊断和完整候选记录。
