# ReplayVLA-P1：最小论文协议与假设矩阵

**状态：APPROVED。2026-09-15 用户批准本协议作为 ReplayVLA-P1 的 scientific scope 增补；旧 `docs/research_spec.md` 的 Δ/R 定义与历史 authority 不回写。**

## 1. 首篇论文唯一主问题

在 frozen VLA 下，过去观测扰动形成的闭环历史后效，与当前/未来观测条件，如何共同影响后续成功与恢复？

首篇论文的核心贡献应是：

1. 经过严格验证的 exact-state branching / replay protocol；
2. 一个普通 success-rate 或即时 action-gap 无法充分解释的 held-out 诊断结果；
3. sham / matched-noise / physics-identity / replay-validity 控制。

Replay infrastructure 本身不是充分论文贡献。Activation exchange 暂不在首篇关键路径；后续若加入，继续遵守旧 research spec 的 paired intervention、no-op、trajectory split 与 Δ/R 定义。

## 2. 四分支定义

在预定切换时刻 `t`，定义完整锚点状态：

`z_t = simulator + controller/wrapper + RNG + policy queue/cache/history + remaining budget`

四个分支：

| Prefix history | Future clean | Future shifted |
|---|---|---|
| Clean | CC | CS |
| Shifted | SC | SS |

同一行的两个 future branches 必须从同一个完整 `z_t` 恢复。跨行状态允许不同，因为 prefix history 本身就是 treatment history。

主比较不允许把 `CC-SC` 直接称为“纯物理 drift effect”；它代表过去 shifted history 留下的总后效，可能同时含物理、controller、queue/cache 等历史差异。

## 3. 观测切换规则

- SC 的 clean observation 必须从 SC 当前物理状态重新渲染；禁止复用 clean rollout 的旧图像。
- Camera intervention 只能改冻结的 observation-side 参数；physics fingerprint 必须保持一致。
- 切换不能推进 physics time、消耗额外动作、改变剩余 horizon，或无意推进全部 observable timers。
- 若需要更新 image cache，只允许更新被协议明确标记的 image-side state，并用 no-restore sham 验证没有额外副作用。

## 4. 切换时刻

主实验使用预定绝对环境动作索引，而不是各 arm 的事件触发时间。建议初始候选：`t ∈ {20, 50, 80}`。

这样避免“某一分支先接触/先抓取，所以更早切换”的 post-treatment selection bias。接触状态可以作为次级分层分析，但不能替代主预注册时间点。

若某一 prefix 在 `t` 前已经 terminal，则该结果作为吸收结果保留；禁止复活环境、补动作或从分母删除该 root。

## 5. Matched noise

显式 action-generation noise 使用冻结 key，例如：

`(root_id, global_env_action_index, draw_kind, draw_slot)`

key 不包含 arm 名称。同一个 root 的可匹配 policy query 使用同一 xi。若某一步不触发 policy query，不凭空消耗一次 noise draw。

不得依赖执行顺序共享 mutable global RNG 来声称 matched-noise。

## 6. 主 estimands

令 `Y` 为原剩余预算内的任务成功指标，越大越好。

定义：

- `G_C = E[Y_CC - Y_CS]`
- `G_S = E[Y_SC - Y_SS]`
- `L   = E[Y_CC - Y_SC]`
- `I   = G_S - G_C`

解释：

- `L`：future clean 下，过去 shifted history 留下的总后效；不是纯物理中介效应。
- `I`：future observation effect 是否依赖过去 history。

H1/H2 为共同主 estimand family；若做显著性检验，使用 Holm 控制两项 family。severity/time 分层必须全部展示，不从多个 cell 中只挑漂亮结果成为主结论。

## 7. 假设矩阵

| ID | 问题 | 证据 | 失败含义 |
|---|---|---|---|
| H0-validity | replay / sham 本身不制造差异 | null、no-switch、no-restore、original-vs-restored | 失败则科学结果 BLOCKED，不解释成模型 robustness |
| H1-history | clean future 是否仍受 shifted history 后效影响 | `L` + task-wise CI | 零效应可有信息；低精度不下强结论 |
| H2-interaction | future observation effect 是否依赖 history | `I`, `G_C`, `G_S` | 不预设方向 |
| H3-diagnostic | action-gap 之外的历史/接触变量是否增加诊断价值 | held-out Brier/log-loss 对比 | 只作次级解释，不当成部署收益 |

H3 不应成为主结果不足时的无限补救路径。

## 8. 数据拆分

建议：

- development tasks：0、4；
- held-out-analysis tasks：1、2、3、5、6、7；
- task 8、9 不作为失败后的自动替代。

这里的 held-out 仅指本研究选择/分析过程，不声称 checkpoint 训练时从未见过这些 tasks。

如果仓库历史已经对某个 validation task 做过结果驱动调参，冻结前必须披露并重划 split，不能继续称严格 held-out。

每个新 task 都要单独做 state-owner / null / replay qualification；task0 的 PASS 不自动外推。

## 9. 扰动与规模

首篇只做一个 camera 轴，例如 agentview yaw。建议 development 内评估 5° / 15° 两个 severity；冻结前不得看 held-out 结果。

建议 validation 默认：

- 6 tasks；
- 每 task 30 roots；
- 2 severity × 3 switch time × 4 branches = 24 cells/root；
- 共 180 roots / 4320 逻辑主分支。

统计独立单位是 root，不是 4320 branches 或大量 frames。

N/task 只允许在 `{20, 30, 40}` 中一次性冻结，且必须在任何 held-out result 可见前决定。选择依据是 development 上的 effect-size/variance、task heterogeneity、compute/storage budget 与预登记 precision target，而不是 p 值。

建议将 `0.10` 的 success-probability difference / interaction 作为需要明确讨论的最小有意义量级；这仍需在 freeze-study 前正式写入 config。

## 10. Controls

主实验前必须具有：

- physical exact-state replay validity；
- policy-state replay validity；
- matched-noise repeatability；
- camera no-op / clean→clean / shifted→shifted sham；
- no-restore direct-switch comparator；
- physics fingerprint invariance；
- technical failure / timeout / missing evidence ledger。

Validation roots 中建议用 deterministic hash 预选 20% 做更完整的 no-restore/sham audit；这些是验证记录，不是额外独立样本。

## 11. 统计

总体估计：先在每个 task 内对 roots 平均，再对 tasks 等权汇总。

CI：按 task 分层、以 root 为 cluster 做 bootstrap；建议 10,000 次、固定 seed。所有 severity/time/arms/noise repeats 同 root 一起重采样。

同时报告：

- 每 task 结果；
- leave-one-task-out；
- technical completion rate；
- full denominator；
- 若有 technical missing，给可计算的上下界/敏感性而不是静默删失。

只有 6 个 tasks 时，不能用海量 frame 数制造虚假的“跨任务高精度”。

## 12. 最小论文实验包

必须包含：

1. runtime/renderer/replay validity 表；
2. 七阶段 same/fresh-process physical replay；
3. policy-state/noise/sham validity；
4. 四分支 `Y`, `G_C`, `G_S`, `L`, `I`；
5. time/severity 图；
6. task-wise + leave-one-task-out；
7. failure/missing evidence 与 compute/storage cost；
8. related-work differential，明确与 robustness benchmark、recovery training、synthetic counterfactual continuation 的差异。

可选：H3 轻量诊断模型、第二 perturbation、第二模型、activation exchange。它们不得成为首篇完成前提。

## 13. 论文主张边界

允许的核心表述应围绕：

- validated exact-state counterfactual branching；
- observation-history 与 future observation 的交互；
- contact-rich / closed-loop history 对恢复的影响；
- failure diagnosis beyond immediate action error（若 H3 真正成立）。

禁止：

- 将 `L` 宣传成纯物理因果效应；
- 把固定 tape replay 当 policy recovery；
- 把一个 task 的 restore PASS 外推到整个 LIBERO；
- 把 trusted third-party runtime 写成已独立证明 native behavior；
- 因为工程 gates 都 PASS 就宣称必然达到 CCF B / SCI 三区。

如果最终只有平凡的“camera shift 降 success”曲线，没有额外诊断信息，则停止扩样本或收窄论文定位，不通过继续加工程复杂度来制造贡献。
