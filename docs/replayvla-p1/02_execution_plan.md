# ReplayVLA-P1：10 日可行性与 6 周执行计划

**状态：APPROVED。与 `01_scientific_admission_contract.md` 共同构成 `xyh/replayvla-p1` 的执行 authority。**

目标不是让旧 G1 PASS，而是在 10 个工作日内得到可信 dynamic replay 的 GO / CONDITIONAL GO / PIVOT 判断；若通过，再争取约 6 周形成首篇可投稿初稿。第 7–8 周只作为有界补证/写作缓冲，不用于无限延长基础设施工作。

## 1. 不做的事情

- 不继续旧 operation-guard 的 E3–E6 或 G2。
- 不训练/微调/LoRA，不引入 world model、SAE 或大规模 head 搜索。
- 不为了吞吐修改 `n_action_steps=1` 的既有动作语义。
- 不原地覆盖旧 runtime、旧证据 bundle、旧 config 或旧失败记录。
- 不把 unit/synthetic test PASS 当成真实 renderer/replay PASS。
- 不因赶时间降低 exact/discrete oracle、null 门槛或 state-closure 要求。

## 2. 复用与新增

优先复用：

- `scripts/m1_state_replay.py` 的 RuntimeAdapter / restore / owner closure；
- `scripts/m1_hard_gate.py` 与 runner 的 scientific oracle；
- `scripts/m1_null_calibration.py` 的 schedule、sidecar 与 comparison logic；
- `scripts/m1_renderer_admission.py` 的 pure validator / EGL adapter；
- `scripts/dcu_preflight.py` 的 CPU environment / DCU policy process seam。

不得恢复已被拒绝的旧 AST/name-heuristic import guard 作为新路线 admission。

建议新增两个薄入口：

- `scripts/m1_sa_workflow.py`：F1–F4 编排；
- `scripts/replayvla_branch_study.py`：F5–F6 四分支与分析。

建议新增 `configs/replayvla/p1.yaml`，但不得简单复制旧 scientific config 后让两份长期漂移。renderer/runtime overlay 必须输出 final effective config hash。

## 3. D1–D10 关键路径

| 时间 | 阶段 | 必须得到的证据 |
|---|---|---|
| D1 | F0 authority / baseline | actual HEAD、dirty/untracked inventory、authority hash、legacy evidence untouched、phase permission state machine |
| D1–D3 | F1 runtime qualification | CPU/DCU 两套解释器与 module origin、完整 traceback、实际 NumPy↔Torch synthetic roundtrip、最多两个隔离候选中的明确结论 |
| D3–D4 | F2 renderer qualification | 强审查后 3 个 fresh workers；每个一次官方构造、一次 public render、close、parent-observed exit；actual EGL identity/ordinal |
| D4–D6 | F3a null | 20 对预注册 duplicate controls（40 attempts），全部失败/完成均保留；physics/RGB 分组 oracle、contact-rich coverage、terminal semantics |
| D6–D8 | F3b physical replay | 七个冻结阶段，每阶段 3 same-process + 3 fresh-process restore，共 42 个窗口验证 |
| D8–D10 | F4 policy replay/noise/sham | 官方 processor/checkpoint、显式 matched noise、queue/cache/RNG closure、至少 free/contact/grasp/carried 的 original-vs-restored policy continuation + camera sham |
| D10 | feasibility decision | 每个 gate 的 PASS/FAIL/BLOCKED、scope、remaining unknowns 与 GO / CONDITIONAL GO / PIVOT |

若 F3 PASS 而 F4 未闭合，必须报告：`PHYSICAL_REPLAY_PASS / POLICY_BRANCHING_BLOCKED`，不得进入论文四分支。

## 4. F0：只做静态与版本化准备

首次进入本地工作区：

```bash
REPO=/public/home/xuyinghao/workspace/vla/ShiftVLA
git -C "$REPO" fetch origin
git -C "$REPO" status --short
git -C "$REPO" rev-parse HEAD
git -C "$REPO" log -1 --format=fuller
```

不得 `git reset --hard`、`git clean` 或 `git add -A`。用户现有 untracked/dirty 文件不得覆盖或自动纳入提交。

若本地分支与远端 authority 存在额外提交，先审 diff；不能直接回退，也不能假设聊天中的 SHA 仍是当前 HEAD。

## 5. F1：runtime qualification

### 5.1 目标

区分并绑定 CPU simulator runtime 与 DCU policy runtime。记录：

- `sys.executable` / Python version；
- NumPy、Torch 的实际 `__file__` / package metadata / wheel origin；
- extension ABI/device 信息；
- CPU/DCU 的实际 host/device transport；
- 导致 `RuntimeError: Numpy is not available` 或 `_ARRAY_API` warning 的完整 traceback 与第一个相关第三方边界。

只使用 synthetic float32 arrays 做 NumPy→Torch、Torch→NumPy、CPU↔DCU 的真实接口 roundtrip。F1 禁止环境构造、真实 init-state 反序列化、EGL、policy/checkpoint 实例化和 rollout。

### 5.2 candidate budget

最多两个隔离候选：当前 runtime baseline + 一个基于明确根因提出的最小 compatibility candidate。禁止无定位地连续升级/降级整套依赖或覆盖旧 venv。

若需要改变 MuJoCo、LeRobot、LIBERO、robosuite scientific semantics、checkpoint 或动作/状态接口，停止并升级路线决策。

**D3 kill gate：**到 D3 若仍没有具体故障定位和可冻结 candidate，不继续 replay 基础设施开发。

## 6. F2：renderer qualification

F2 之前必须进行一次强审查，范围仅限新 M1-SA-v1，而不是重新要求旧 G1 native closure。审查至少覆盖：

- entrypoint 和 phase gating；
- source/install identity；
- legacy configs/bundles 不被改写；
- scientific oracle 未降低；
- no-overwrite / failure publication；
- native runtime 被明确标注为 trusted dependency，而非 observed-zero。

随后允许三个 fresh workers。每个 worker：

1. 使用官方环境路径；
2. 允许一次官方 construction reset 及其规定的内部 init/reset/settle；
3. 恰好一次 public render；
4. close；
5. parent 观察正常/异常退出并保存 stdout/stderr。

旧 ordinal `8` 不能直接继承；历史 CPU ordinal `0` 也只是候选，不是当前 PASS。禁止循环试设备直到成功后只记录成功值。

F2 仍禁止 policy/processor 调用、研究 step、replay 和 null schedule。

## 7. F3：null + physical replay

### 7.1 Null

在 F2 PASS 后，先冻结新 null schedule，然后执行至少 20 对 complete duplicate controls。保留：

- 每个 attempt 的 terminal record；
- 失败、超时、技术缺失；
- physics 与 renderer 分组 oracle；
- contact/grasp/carried 的独立正证据；
- discrete exact gates。

不得从旧 ordinal-8 失败 bundle 继承新 null PASS。

### 7.2 Physical replay

优先继续使用已冻结 task0/init0/seed2027 与 82×7 action tape，但必须重新验证实际文件/bytes/hash 与 candidate runtime 的适用性；不能只比较 caller-provided 常量。

七个阶段全部保留：

- free motion: capture 1, horizon 10
- pre-contact: 33, 10
- contact: 43, 10
- grasp: 49, 10
- carried: 54, 10
- release: 80, 2
- predicate transition: 81, 1

每个阶段：3 same-process + 3 fresh-process restore。restore 以后禁止 reset、set-init、settle、dummy action、autoreset、retry、物理模型 patch。

动作字节、contact identity、predicates、termination、counters、gripper discrete state 必须 exact；浮点只使用冻结的 null-group oracle，不增加 global epsilon 或 post-hoc multiplier。

若 candidate runtime 使原 source cohort/model fingerprint 不再适用，必须显式产生新 versioned cohort；不得改旧 tape/config 冒充同一实验。超出 D10 预算则停止。

## 8. F4：policy-state replay / matched noise / sham

Physical replay PASS 以后，才允许固定官方 checkpoint/processor 与 policy continuation。

Capture/restore 必须处理或证明：

- 未执行 action queue 与 cursor；
- policy/processor 有效 cache/history；
- Python / NumPy / Torch CPU / DCU RNG，或显式 action-generation noise；
- instruction / proprioception / remaining budget；
- device 与数值路径。

不能因为 `n_action_steps=1` 就假设 policy 无状态。

先验证同 observation + same noise 的 policy repeatability，再验证 original-vs-restored no-intervention continuation。至少覆盖 free/contact/grasp/carried，每类 3 same + 3 fresh 或合法同步 terminal。

Camera sham / clean→clean / shifted→shifted control 不能推进 physics time、消费额外 action、改变 remaining budget 或无意推进整个 observable tree。允许更新明确指定的 image cache，但必须与 no-restore comparator 一致。

Fixed action tape 只能证明 physical replay，不能证明 policy recovery。

## 9. 需要新增/映射的行为测试

至少覆盖：

- 没有前置 PASS manifest 就拒绝后续 phase；
- actual bytes 漂移即使 caller hash 正确也拒绝；
- 新 renderer overlay 不改旧 ordinal-8 config；
- 失败/超时必须 terminalize 为 FAIL/BLOCKED；
- attempt/output/schedule duplicate 在 entry 前拒绝；
- unknown mutable owner 或 queue/cache 漏存失败；
- restore 后 reset/settle/dummy/autoreset 失败；
- null group 缺失、physics/RGB oracle 混用失败；
- same noise + same input 的 no-op policy replay 可复现；
- camera sham physics fingerprint 不变且不推进 queue/time；
- switch 前 terminal 作为吸收结果，不复活；
- root 全部 arms/severity/time/noise 属于同一个 split/bootstrap cluster；
- 统计单位是 root，不把 branch/frame 当独立样本；
- held-out 冻结后改变 N/task/severity/metric 必须新版本而非覆盖。

Synthetic tests 只证明 harness logic，不证明真实 native/runtime/replay。

## 10. 建议的新 CLI 契约

以下接口是计划目标，Codex 需要实现后才能执行：

```bash
"$CPU_PY" -B scripts/m1_sa_workflow.py qualify-runtime --config "$CFG" --output-root "$ROOT/runtime"
"$CPU_PY" -B scripts/m1_sa_workflow.py qualify-renderer --config "$CFG" --output-root "$ROOT/renderer"
"$CPU_PY" -B scripts/m1_sa_workflow.py prepare-null --config "$CFG" --output-root "$ROOT/null"
"$CPU_PY" -B scripts/m1_sa_workflow.py run-null --config "$CFG" --output-root "$ROOT/null"
"$CPU_PY" -B scripts/m1_sa_workflow.py verify-physical-replay --config "$CFG" --output-root "$ROOT/physical"
"$CPU_PY" -B scripts/m1_sa_workflow.py verify-policy-replay --config "$CFG" --output-root "$ROOT/policy"
"$CPU_PY" -B scripts/replayvla_branch_study.py pilot --config "$CFG" --output-root "$ROOT/development"
"$CPU_PY" -B scripts/replayvla_branch_study.py freeze-study --config "$CFG" --output-root "$ROOT/study"
"$CPU_PY" -B scripts/replayvla_branch_study.py run-study --config "$CFG" --output-root "$ROOT/study"
"$CPU_PY" -B scripts/replayvla_branch_study.py analyze --config "$CFG" --output-root "$ROOT/study"
```

每个入口必须验证 contract hash、execution commit、parent gate、actual input hashes 和 no-overwrite。不得用旧 hard-gate command 仅改环境变量冒充新 SA cohort。

## 11. D10 之后

只有 F1–F4 都在声明 scope 内 PASS，才进入 F5/F6。

- Week 3：development pilot、task qualification、吞吐/存储/precision estimate、novelty differential。
- Week 4：冻结 task/severity/switch/N/analysis，生成 held-out schedule。
- Week 5：一次完整 held-out 主实验；所有 preregistered rows 均有状态，不补跑失败 seed 凑显著。
- Week 6：主图表、统计、局限、复现材料、初稿。

Activation exchange 默认不在首篇关键路径；主行为实验完成后若预算充足才作为扩展。

## 12. 完成定义

每阶段 evidence bundle 至少包含：input manifest、argv/command、runtime identity、stdout/stderr、case-level outcomes、hash-verified sidecars、coverage limits 与 terminal judgment。

最终必须能追溯：

- old/new contract mapping；
- SA-runtime / renderer / null / physical / policy 各 gate；
- paper preregistration 与全 root disposition；
- clustered statistical analysis；
- failure/missing evidence；
- exact code/config/data references。

Commit 数量、unit test 数量或 agent 声称完成，均不能替代 scientific gate PASS。
