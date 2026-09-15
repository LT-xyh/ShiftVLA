# Codex 执行交接

**状态：APPROVED EXECUTION HANDOFF。**

Repository: `LT-xyh/ShiftVLA`
Branch: `xyh/replayvla-p1`
Baseline ancestor: `e37dfa93dfcc5793340829f8c685a4f5ec8c172b`

GitHub 中 `docs/replayvla-p1/` 是新路线 authority。开始工作前从 Git 重建状态，不依赖旧聊天隐含上下文。

## 1. 必须先做的同步

```bash
cd /public/home/xuyinghao/workspace/vla/ShiftVLA
git fetch origin
git status --short
git rev-parse HEAD
git checkout xyh/replayvla-p1
git pull --ff-only origin xyh/replayvla-p1
```

如果 checkout/pull 会覆盖本地 dirty/untracked 文件，不要 reset/clean。先记录冲突与本地状态，然后采用不丢失用户文件的方式处理。

不得自动提交既有 `AGENTS.md` 修改、运行产物或无关 untracked files。不得 `git add -A`。

## 2. Authority / 历史边界

保留：

- E2 = `COVERED / PASS (E2-local)`；
- 旧 G1 = `BLOCKED`；
- 旧 E3–E6 workflow = `NOT STARTED`；
- known predicates 中此前记录的 unresolved 不回写成 PASS；
- G2 = `NOT AUTHORIZED`；
- 旧 Renderer Preflight / M1-N0 结果原样保留。

暂停旧 operation-guard 支线。不要继续旧 E3–E6/G2，也不要把新路线任何 PASS 回填成旧 G1 PASS。

## 3. 执行顺序

严格：

`F0 authority/baseline → F1 runtime → F2 renderer → F3 null + physical replay → F4 policy/noise/sham → F5 development → F6 frozen held-out study`

后续 phase 只有在前置真实 PASS 后才自动获得权限。整体批准不是无限运行授权。

阶段内允许自行：实现、synthetic tests、最小 bug 修复、验证、写 evidence、精确 commit、push。无需每个小 checkpoint 请求人工确认。

触发以下情况才停止并升级：

- D3 / D10 / Week-3 kill gate；
- 需要改 scientific oracle、task/model definition、trust boundary；
- 需要超过批准 runtime candidate/cohort budget；
- 需要新 native probe 或重开旧 E3/G2；
- evidence contamination；
- unknown scientific mutable state 无法闭合；
- 超出时间/存储/计算预算。

## 4. 第一阶段：F0 + F1

当前优先目标不是实现全部 replay pipeline，而是完成 runtime qualification。

### F0

核验：

- actual local/remote HEAD；
- dirty/untracked；
- authority docs；
- old configs/runs/evidence 未被修改；
- 新 phase state machine 与 evidence namespace。

可以添加新 schema/config/tests，但 F0 禁止环境、EGL、policy、schedule。

### F1

分别检查 CPU simulator runtime 与 DCU policy runtime。

必须产出：

- `sys.executable` / Python version；
- NumPy/Torch 实际 module origin / package/wheel identity；
- 完整 traceback；
- synthetic float32 NumPy→Torch、Torch→NumPy、必要 host/device bridge；
- 当前 candidate 的 PASS/FAIL/BLOCKED；
- 若需要，只允许一个额外隔离 compatibility candidate（总计最多两个 candidate）。

F1 禁止：

- 环境构造；
- task init-state 真反序列化；
- EGL/context/render；
- policy/checkpoint 实例化；
- scientific rollout；
- 盲目整套升级/降级依赖；
- 原地覆盖旧 venv。

D3 若仍没有具体根因/具体 failing boundary 与合格 candidate，停止并报告，不继续换包试运气。

## 5. F2 前强审查

在首次真实 renderer worker 运行前，先完成一次高强度 review。可以由主 Codex agent 执行；独立 reviewer 不可用时必须明确记录 unavailable，不得伪造 review PASS。

检查：

- new phase gate 不能越权；
- old config/run/evidence 不覆盖；
- scientific oracle 无降低；
- actual bytes/input binding；
- source/install/runtime identity；
- failure/no-overwrite behavior；
- native runtime 被明确视为 `TRUSTED_DEPENDENCY`，而非 observed-zero。

通过后才进入 F2。

## 6. F2–F4 摘要

### F2 renderer

3 fresh workers；每个一次官方 construction reset/内部规定初始化、一次 public render、close、parent-observed exit。记录 actual EGL identity/ordinal 与 stdout/stderr。

禁止循环尝试所有 EGL 设备只保存成功结果。

### F3 null + physical replay

先冻结新 null schedule；至少 20 对 duplicate controls。随后 task0/init0/seed2027 的七阶段 replay：

- free 1/10
- pre-contact 33/10
- contact 43/10
- grasp 49/10
- carried 54/10
- release 80/2
- predicate transition 81/1

每阶段 3 same + 3 fresh。restore 后禁止 reset/set-init/settle/dummy/autoreset/retry/model patch。

### F4 policy/noise/sham

只有 F3 PASS 后进入。使用 frozen official checkpoint/processor；显式 matched noise；恢复/证明 queue/cache/RNG/remaining budget。至少 free/contact/grasp/carried 做 original-vs-restored policy continuation 与 camera sham。

Physical replay PASS 但 F4 不通过时，状态必须是：

`PHYSICAL_REPLAY_PASS / POLICY_BRANCHING_BLOCKED`

不得进入 F5。

## 7. 代码与证据原则

- 尽量复用已有 RuntimeAdapter / scientific oracle / null logic。
- 不恢复旧 name-based/AST import guard。
- 不把 unknown owner 改成 ignored。
- 不创建新的 global epsilon 或 post-hoc tolerance。
- 不用 fixed action tape 声称 policy recovery。
- 不用 clean rollout 的旧图像给 shifted physical state 作为 clean future observation。
- 不隐藏 FAIL/BLOCKED/timeout/missing。
- 真实 experiment attempt 失败后不覆盖或无修改重试。
- 工程 test 可以在同阶段最小修复后重跑。

## 8. Git 工作方式

每个有意义阶段：

1. 只 stage 本阶段相关文件；
2. 跑对应 tests / verification；
3. `git diff --check`；
4. commit；
5. push `xyh/replayvla-p1`；
6. 返回 commit SHA、执行命令、结果、evidence paths、当前 gate 与限制。

不要把大量 stdout 复制到聊天；将 evidence 落仓库允许的文档/manifest 或 runs artifact 中，并在汇报里给路径和摘要。

## 9. D10 汇报格式

只需一次路线级汇报：

- exact HEAD；
- F1/F2/F3/F4 状态；
- runtime candidate 与实际环境；
- physical replay 适用域；
- policy replay 适用域；
- failures / missing / unavailable reviewers；
- consumed time/compute/storage；
- GO / CONDITIONAL GO / PIVOT；
- 若继续，是否允许 F5 development pilot。

不要在正常执行中再次建议“重新开 Pro 规划”。已批准计划是默认 authority，除非触发上述停止条件。
