# Findings & Decisions

## Requirements
- 所有可跨空间指标必须按老师的冻结八设置完整实验。
- RB40 必须 40/40，不能用 29/40 得出最终结论。
- NB201 必须覆盖 e0/e5/e10/e25/e50，每阶段 64 个架构。
- DARTS 必须覆盖 e0/e100，每阶段 100 个架构。
- 缺失 setting 不得记零、删除或参与 All-setting mean。
- 设计更多有机制依据的新指标，保留全部候选与负结果。
- 主线新指标采用可解释、无拟合权重的公式；拟合型组合只能作为单独标注的附加消融。

## Research Findings
- 冻结协议：RB40 mature、NB201 五阶段、DARTS 两阶段，共八设置。
- 固定测量参数：64样本、offset=0、seed=20260808、epsilon=8/255。
- E36 原始数据为4,480行、4,480唯一键、全部status=ok且数值有限。
- E36覆盖：RB40 40×8、NB201 64×5×8、DARTS 100×2×8。
- E36 correlations.csv 有64个“指标×设置”相关性单元，全部有效，bootstrap=2,000。
- 历史 existing_metrics_full_v2 有38,080行和68个指标，但只有36,296行成功、1,784行失败。
- 58个历史指标在RB40恰好缺11个模型，即只使用29/40；另有4个缺5个、3个缺2个。
- 两个RACL代理分别只适用于特定空间，不能强行作为同一跨空间指标；importance指标当前全网格非有限。
- v2严格分析已将68个指标全部标为incomplete且不给All-setting mean；该结论只描述v2原始运行。
- 后续发现 `existing_metrics_full_v3_cpu_repair` 已补回661条：36,957成功、1,123失败，62个历史指标严格完整。
- v3剩余：两个RACL空间专用共560条、importance全网格560条、三个TRNAS在Kang2021Stable各失败1条。
- 当前70指标comparison表实际由v3的62个完整历史指标加8个完整E36指标组成；62项分数逐项一致，不含29/40结果。此前关于该comparison表绕过v2状态的判断已撤回。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 先修复现有指标，再设计新指标 | 防止用不完整基线虚高新指标名次 |
| 空间专用指标进入附表，不进入跨空间总榜 | 数学定义不一致时强行填值会失去可比性 |
| 新指标冻结后只进行一次正式留出评测 | 降低榜单过拟合与事后调参 |
| 无拟合权重作为主线，拟合组合仅作消融 | 用户于2026-09-29明确同意 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 起初只审计v2，遗漏服务器已有v3修复运行 | 已对照v3与comparison逐项验证；当前70项均完整，但还不是所有注册指标的最终表 |

## Resources
- `docs/superpowers/specs/2026-09-17-three-space-benchmark-design.md`
- 服务器：`/workspace/robust_metric_lab_nas/outputs/student_metrics/wangjiahao/`
- E36：`e36_full_metric_design_v1/`
- 历史全指标：`existing_metrics_full_v2/`

## Visual/Browser Findings
- 用户截图显示当时GPU和CPU无实验进程；原因是E36补跑已经结束，并非仍在运行。
