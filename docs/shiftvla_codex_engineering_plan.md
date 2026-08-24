可以。这个项目很适合用 Codex，但要把 Codex 当作“实验工程师”，而不是让它自由决定研究方案。你现在的科学方案已经冻结，Codex 的任务是把它**可靠地工程化、逐步测试、记录可复现实验**。核心仍然以最终的 ShiftVLA 路线为准：Frozen SmolVLA → matched state → matched sampling → paired activation exchange → (\Delta) → held-out restoration。

我建议直接按下面这个开发流程执行。

# 一、不要先对 Codex 说“实现 ShiftVLA”

这是最容易失败的做法。

如果直接输入：

> 帮我实现 ShiftVLA，复现所有实验。

Codex 很可能一次改十几个文件，同时混进：

* 自己理解的 activation hook；
* 自己设计的 state replay；
* 自己设计的 metric；
* 随机的目录结构；
* 没验证过的 LIBERO API；
* 甚至顺手开始训练模型。

最后很难判断实验错在模型、环境还是 intervention。

正确方式是：

[
\boxed{
\text{Research Spec}
\rightarrow
\text{Architecture Audit}
\rightarrow
\text{Milestone-by-Milestone Implementation}
}
]

而且每个 milestone 都必须满足：

[
\boxed{\text{test → implementation → verification → commit}}
]

再进入下一步。

OpenAI 当前也推荐使用 `AGENTS.md` 给 Codex 提供持久项目级约束；Codex App/CLI 的 `/init` 可以生成初始 `AGENTS.md`。([OpenAI][1])

---

# 二、项目目录建议这样建

不要直接在 LeRobot 仓库里堆实验脚本。

建议：

```text
shiftvla/
├── AGENTS.md
├── README.md
├── pyproject.toml
│
├── docs/
│   ├── research_spec.md
│   ├── experiment_protocol.md
│   ├── architecture_map.md
│   └── decisions/
│
├── configs/
│   ├── baseline.yaml
│   ├── camera_smoke.yaml
│   └── camera_pilot.yaml
│
├── src/shiftvla/
│   ├── envs/
│   │   ├── state_replay.py
│   │   └── matched_render.py
│   ├── policies/
│   │   └── smolvla_adapter.py
│   ├── interventions/
│   │   ├── controller.py
│   │   ├── groups.py
│   │   └── cache.py
│   ├── metrics/
│   │   ├── dependency.py
│   │   └── restoration.py
│   ├── data/
│   │   ├── schema.py
│   │   └── split.py
│   └── utils/
│       ├── reproducibility.py
│       └── manifest.py
│
├── scripts/
│   ├── 00_env_audit.py
│   ├── 01_eval_baseline.py
│   ├── 02_build_matched_pairs.py
│   ├── 03_smoke_intervention.py
│   ├── 04_discover_delta.py
│   └── 05_validate_restoration.py
│
├── tests/
│   ├── test_state_replay.py
│   ├── test_matched_sampling.py
│   ├── test_intervention_noop.py
│   ├── test_activation_exchange.py
│   └── test_metrics.py
│
├── outputs/
└── external/
    ├── lerobot/
    └── LIBERO-plus/
```

`outputs/` 不进 Git，代码、配置、测试、统计脚本全部进 Git。

---

# 三、先写 `AGENTS.md`

这是你使用 Codex 最重要的一步。

把下面这份直接放进项目根目录：

```markdown
# ShiftVLA Engineering Instructions

## Project mission

Implement the frozen-policy ShiftVLA pilot for testing whether
observation distribution shifts reconfigure internal dependencies
of Vision-Language-Action policies.

The scientific design is defined in:
- docs/research_spec.md
- docs/experiment_protocol.md

Do not alter scientific definitions without explicit human approval.

## Pilot scope

The pilot uses:

- Model: frozen SmolVLA
- Benchmark: LIBERO / LIBERO-Plus
- First perturbation: Camera
- Intervention granularity: 6-8 architecture-aware groups
- No model training in the pilot
- No LoRA
- No World Model
- No SAE
- No exhaustive head-level analysis

## Scientific invariants

Matched comparisons MUST keep fixed:

- underlying simulator state
- task instruction
- proprioceptive state
- model checkpoint
- preprocessing
- action-generation noise / flow initial condition

Only the observation perturbation may change.

Discovery and validation data MUST be split by trajectory,
never by random frames.

## Intervention rules

Primary intervention:
paired clean/OOD activation exchange.

Zero/mean ablation is only a robustness/sanity check.

Counterfactual restoration is an analysis tool,
not a deployable repair algorithm.

## Engineering rules

1. Never silently modify third-party behavior.
2. Prefer wrappers and explicit instrumentation.
3. Any necessary LeRobot modification must be minimal,
   backward-compatible, disabled by default, and tested.
4. Do not bypass official LeRobot preprocessors.
5. Every experiment must produce a run manifest.
6. Every script must accept an explicit seed and config path.
7. Do not hard-code task IDs, paths, devices, or checkpoints.
8. Never use discovery data to evaluate held-out restoration.
9. Do not report a result until its verification command has run.
10. Keep changes small and commit after each verified milestone.

## Reproducibility

Every run manifest must include:

- git SHA
- LeRobot git SHA
- LIBERO / LIBERO-Plus git SHA
- checkpoint identifier and revision
- Python version
- PyTorch version
- transformers version
- CUDA version
- GPU model
- seed
- task IDs
- trajectory IDs
- perturbation settings
- action-generation noise seed
- config file
```

这会极大降低 Codex“越写越偏”的概率。

---

# 四、再写 `research_spec.md`

这个文件只放**科学定义**，Codex 不允许自行修改。

至少写清楚：

[
a_{ID}=\pi(o^{ID};\xi)
]

[
a_k=\pi(o^k;\xi)
]

其中：

[
s,\ instruction,\ proprioception,\xi
]

全部 matched。

以及：

[
I_j^k(s)
========

D(
\pi^{do(j)}(o_s^k;\xi),
a_{ID}
)
-

D(
\pi(o_s^k;\xi),
a_{ID}
)
]

[
\delta_j^k(s)
=============

I_j^k(s)-I_j^{ID}(s)
]

[
\boxed{
\Delta_j^k
==========

\mathbb E_s[\delta_j^k(s)]
}
]

restoration：

[
\boxed{
R_j^k(s)
========

\frac{
D(a_k,a_{ID})
-------------

D(a_k^{patch(j)},a_{ID})
}{
D(a_k,a_{ID})+\epsilon
}
}
]

并注明：

> demo action distance 是辅助指标；closed-loop success 是最终 behavioral validation。

这样以后 Codex 不会突然把你的指标改成 MSE against demonstration。

---

# 五、第一项 Codex 任务不是写代码，而是“读代码”

这是我非常建议你执行的一步。

在项目根目录运行 Codex 后，第一条 prompt：

```text
Read AGENTS.md and docs/research_spec.md first.

Do not modify implementation code yet.

Audit the currently pinned LeRobot SmolVLA implementation and trace the
complete inference path from the public policy action-selection API to
the final action chunk.

Specifically inspect:

1. SmolVLAPolicy
2. VLAFlowMatching.sample_actions
3. SmolVLMWithExpertModel
4. image/VLM prefix computation
5. action expert computation
6. flow-matching denoising/integration
7. KV-cache behavior

Determine:

- where action-generation noise is created and whether it can be supplied explicitly;
- which exact modules correspond to vision encoder, connector, VLM layers,
  action expert layers and action projections;
- whether normal PyTorch forward hooks on transformer blocks actually fire;
- the safest locations for paired activation exchange;
- tensor shapes at each proposed intervention location;
- whether intervention is applied once during prefix computation or repeatedly
  during flow denoising.

Write the result to docs/architecture_map.md.

Do not implement the intervention yet.
```

这一步非常重要。

因为当前 SmolVLA 并不是一个简单：

```python
for layer in model.layers:
    x = layer(x)
```

的网络。

当前 LeRobot 实现中的 `SmolVLMWithExpertModel.forward()` **自己手工展开了 VLM 和 expert layer 的 attention、residual、MLP 计算**。因此你如果让 Codex 天真地：

```python
model.layers[i].register_forward_hook(...)
```

很可能 hook 根本不是你以为的那个 block output。([GitHub][2])

所以一定：

> **先让 Codex 画调用图，再决定 hook seam。**

---

# 六、一个好消息：matched action noise 很容易实现

当前 SmolVLA 的 `sample_actions()` 已经有：

```python
noise=None
```

参数。

如果不传才内部：

```python
noise = self.sample_noise(...)
```

然后从该 noise 开始 Euler flow integration。([GitHub][3])

因此你不需要靠：

```python
torch.manual_seed(...)
```

碰运气。

应该直接：

```text
sample one explicit noise tensor ξ
```

然后：

```text
ID forward     -> noise = ξ
OOD forward    -> noise = ξ
patched OOD    -> noise = ξ
patched ID     -> noise = ξ
```

这是 Codex 第二阶段必须实现的 invariant。

测试应该要求：

```text
same observation + same explicit noise
→ same action
```

在合理浮点误差范围内成立。

---

# 七、接下来按 6 个 milestone 让 Codex 工作

| Milestone | Codex 做什么                     | 通过条件                                 |
| --------- | ----------------------------- | ------------------------------------ |
| M0        | 环境审计 + baseline               | policy 可以稳定完成官方 eval                 |
| M1        | simulator state replay        | 相同 state 可重复恢复                       |
| M2        | matched ID/OOD rendering      | state 完全一致，仅图像变化                     |
| M3        | deterministic SmolVLA wrapper | 同 state + 同 obs + 同 noise → 同 action |
| M4        | activation exchange           | disabled intervention 与原模型输出一致       |
| M5        | (\Delta) discovery            | 20→200 states 能稳定生成结果                |
| M6        | held-out restoration          | discovery Top-K 在独立 trajectories 测 R |

每个 milestone **单独给 Codex prompt**。

不要一次布置 M0–M6。

---

# 八、M0：先复现 baseline

你可以给 Codex：

```text
Implement Milestone M0 only.

Goal:
Create a reproducible frozen SmolVLA LIBERO baseline evaluation harness.

Requirements:

- Do not train or fine-tune anything.
- Use lerobot/smolvla_libero.
- Use official LeRobot preprocessing and environment processors.
- Pin and record all relevant package/repository versions.
- Save per-episode success, task ID, seed and rollout length.
- Save run_manifest.json.
- Add a smoke configuration using only 1-2 tasks and 1 episode.
- Add a full configuration separately.
- Do not proceed to state replay.

Before implementation:
1. inspect the existing APIs,
2. write a short implementation plan,
3. identify tests,
4. then implement.

Completion requires running the smoke test and showing its output.
```

当前 Hugging Face 确实有官方 `lerobot/smolvla_libero` checkpoint。([Hugging Face][4])

而且一定要求 Codex **使用 LeRobot 官方 processor**。

因为当前 LIBERO processor 包含一个很容易漏掉的细节：

> 数据集图像与 raw LIBERO simulator 存在 180° convention difference，processor 会进行相应处理。([GitHub][5])

如果 Codex 自己：

```python
obs["agentview_image"] -> torch.tensor -> policy
```

baseline 很可能直接被搞坏。

---

# 九、M1：state replay

完成 baseline 才让 Codex做：

```text
Implement M1: deterministic LIBERO state replay.

Do not implement perturbations or activation interventions yet.

First inspect the underlying LIBERO / MuJoCo state representation.

Requirements:

- Define a serializable MatchedStateRecord.
- Support restoring a simulator state.
- Preserve task identity and instruction.
- Verify robot state and object state after restore.
- Verify repeated restoration produces equivalent observations.
- Add tests comparing physical state before save and after restore.
- Distinguish initialization-state replay from arbitrary mid-trajectory state replay.
- Do not assume set_init_state is sufficient for arbitrary trajectory states;
  inspect the underlying simulator API and demonstrate correctness.

Produce a smoke test using several states from one trajectory.
```

为什么我要特别加最后一句？

因为当前 LeRobot wrapper 的 `reset()` 确实调用：

```python
self._env.set_init_state(...)
```

然后还会执行数个 no-op step 让物体稳定。([GitHub][6])

但是：

> **“可以恢复 benchmark 初始状态”不等于“已经证明可以无误恢复任意 trajectory 中间状态”。**

这必须让 Codex 用测试证明，而不是猜。

---

# 十、M2：matched Camera pair

通过 state replay 后才做 Camera。

Codex prompt 核心：

```text
Implement M2: matched ID/Camera observation pairs.

For every pair, require:

physical_state_ID == physical_state_OOD
instruction_ID == instruction_OOD
proprioception_ID == proprioception_OOD

Only camera parameters may differ.

Add an automated invariant checker that fails the run if any
non-camera state field differs.

Record camera perturbation parameters and seeds.

For the initial smoke test:
- one task
- 20 states
- one camera perturbation severity

Do not implement model interventions yet.
```

LIBERO-Plus 当前官方 LeRobot 文档提供 Camera、Lighting、Background、Sensor Noise 等七类扰动，且 `libero_plus` 使用和 LIBERO 相近的接口。([GitHub][7])

但注意一个环境坑：

> LIBERO-Plus 安装会替换 vanilla `libero` Python package，两者不能在同一个 Python 环境中正常共存。([GitHub][7])

因此我建议专门建：

```text
shiftvla-libero-plus
```

环境。

不要在一个 conda env 里来回：

```bash
pip uninstall libero
pip install ...
pip uninstall ...
```

很容易得到不可复现实验环境。

服务器/headless 环境设置：

```bash
export MUJOCO_GL=egl
```

也是官方当前 LIBERO-Plus 的推荐配置。([GitHub][7])

---

# 十一、M3：先做 matched sampling，不碰 activation

要求 Codex 实现一个很薄的：

```python
SmolVLAMatchedRunner
```

接口类似：

```python
result = runner.run(
    observation=obs,
    noise=noise,
)
```

输出：

```python
MatchedActionResult(
    action_chunk=...,
    noise=...,
)
```

最重要的测试：

```text
obs A + noise ξ -> a1
obs A + noise ξ -> a2
```

要求：

[
a_1\approx a_2.
]

然后：

```text
ID obs + ξ
OOD obs + ξ
```

才计算：

[
D(a_{OOD},a_{ID}).
]

这里不要先做任何 activation patch。

---

# 十二、M4：Activation Exchange 是最需要谨慎的部分

我建议让 Codex **先做一个 instrumentation design，再实现**。

目标是给 LeRobot 增加一个默认关闭的 callback，例如概念上：

```python
hidden = intervention.apply(
    stream="vlm",
    layer_idx=i,
    phase="prefix",
    hidden=hidden,
)
```

以及：

```python
hidden = intervention.apply(
    stream="expert",
    layer_idx=i,
    phase="denoise",
    timestep=t,
    hidden=hidden,
)
```

但是具体接口应该先让 Codex 根据当前源码设计。

最关键的测试不是：

> patch 能不能改变输出。

而是：

### No-op equivalence

```text
original model
```

和：

```text
instrumented model + disabled controller
```

必须满足：

[
a_{\text{original}}
\approx
a_{\text{instrumented,no-op}}.
]

如果这项测试不过：

[
\boxed{\text{禁止进行论文实验}}
]

因为这说明 instrumentation 本身已经改变了模型。

---

# 十三、Pilot 先不要缓存全部 activation

不要让 Codex生成：

```text
2000 states
× 32 layers
× all tokens
× all denoise steps
```

的巨大 `.pt` 文件。

正确流程应该是：

```text
load one matched state
→ clean forward/cache needed activation
→ OOD forward
→ patched OOD forward
→ compute metrics
→ discard large tensors
→ save scalar results
```

长期只存：

```text
sample_id
task_id
trajectory_id
step_id
shift
severity
component
D_clean_ood
intervention_effect
delta
restoration
seed
noise_seed
```

以及少量 debugging samples。

这样存储不会爆炸。

---

# 十四、M5 才计算 (\Delta)

Codex prompt：

```text
Implement M5: Camera dependency discovery.

Use only discovery trajectories.

Do not use validation trajectories for any ranking or threshold choice.

For each matched state and component group compute the exact quantities
defined in docs/research_spec.md.

Save state-level results in a tidy table.

Aggregation must preserve:
task_id
trajectory_id
state_id
component
condition

Implement hierarchical bootstrap:
task -> trajectory -> state.

Generate:
1. component-wise Delta estimates
2. 95% confidence intervals
3. component heterogeneity test
4. a diagnostic plot

Do not select final Top-K using validation data.
```

第一轮只：

[
20\text{ states}\times8\text{ groups}
]

测试代码。

通过后：

[
200+\text{ states}.
]

---

# 十五、M6 才做 held-out restoration

再给 Codex：

```text
Implement M6: held-out restoration validation.

Inputs:
- frozen component ranking produced from discovery data
- held-out trajectories only

Compare:
Top-Delta-K
Random-K
ID-important-K

The selected component set must be frozen before loading validation results.

Compute normalized restoration R exactly as defined in research_spec.md.

Use cluster/hierarchical bootstrap over trajectories.

The primary comparison is:
R(Top-Delta-K) - R(Random-K)

Do not add Lighting, Noise, LoRA, or a second model.
```

这里如果出现：

[
TopK(\Delta)>RandomK
]

才正式进入下一阶段。

---

# 十六、Codex 的 commit 节奏也固定下来

我建议每完成一个 milestone 就形成一个 commit/tag：

```text
m0-baseline
m1-state-replay
m2-matched-camera
m3-matched-sampling
m4-intervention
m5-camera-delta
m6-heldout-restoration
```

特别不要让 Codex做：

```text
"Refactored everything + added experiment"
```

这种 4000 行大 commit。

论文实验代码最怕这种开发方式。

---

# 十七、每次给 Codex 的任务都加这句话

这是非常有用的 prompt 尾巴：

```text
Before claiming completion:

1. run the relevant tests;
2. run the smallest end-to-end smoke experiment;
3. inspect the generated output;
4. report exactly which commands were executed;
5. report any assumptions that remain unverified;
6. show git diff --stat;
7. do not proceed to the next milestone.
```

这会大幅减少：

> “代码看起来已经完成，但实际上从未执行。”

的问题。

---

# 十八、今天实际应该做什么

如果今天正式开始，我建议**不要让 Codex写 activation patch**。

今天只完成这三个东西：

```text
AGENTS.md
docs/research_spec.md
docs/architecture_map.md
```

然后：

[
\boxed{\text{M0 baseline}}
]

今天给 Codex 的第一条完整任务，我建议直接复制下面这段：

```text
We are starting the ShiftVLA project.

First read:
- AGENTS.md
- docs/research_spec.md

This is a frozen-policy mechanistic experiment. Do not train,
fine-tune, or change the scientific definitions.

Your first task is READ-ONLY architecture and environment reconnaissance.

Inspect the pinned LeRobot and LIBERO/LIBERO-Plus source code.

Trace:

1. how the official SmolVLA LIBERO checkpoint is loaded;
2. the complete observation preprocessing pipeline;
3. the complete SmolVLA inference path;
4. where flow-matching action noise is sampled and how to inject explicit noise;
5. the VLM, vision encoder, connector and action-expert architecture;
6. the exact transformer execution loop and whether normal forward hooks are valid;
7. LIBERO reset/state APIs and how exact MuJoCo states can be serialized/restored;
8. how LIBERO-Plus camera perturbations are represented.

Write:
docs/architecture_map.md

The document must contain:
- exact source file paths;
- relevant classes/functions;
- call graph;
- important tensor shapes;
- proposed state-replay seam;
- proposed activation-intervention seam;
- known risks;
- a proposed architecture-aware 6-8 component grouping.

Do not modify runtime code yet.

At the end, propose the implementation design for M0 baseline only.
Stop and wait for approval before implementing M0.
```

**这一条做完，把 Codex 生成的 `architecture_map.md` 和它提出的 M0 方案发给我看。**

我最想先帮你检查两个地方：

[
\boxed{\text{state replay 是否真的正确}}
]

以及：

[
\boxed{\text{activation hook seam 是否选对}}
]

因为这两个如果最开始做错，后面即使跑出很漂亮的 (\Delta) 图，也可能全部是伪结果。

你现在已经把 Codex/WSL/服务器链路跑通了，因此工程上最合理的是：**Codex CLI 在 Linux/GPU 服务器项目根目录负责实现和测试，Windows/WSL 主要负责控制、审阅 diff 和整理实验；MuJoCo headless 统一走 EGL。** ([OpenAI Help Center][8])

[1]: https://openai.com/zh-Hans-CN/index/introducing-codex/?utm_source=chatgpt.com "Codex 简介 | OpenAI"
[2]: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/smolvlm_with_expert.py?utm_source=chatgpt.com "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py at main · huggingface/lerobot · GitHub"
[3]: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py "lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py at main · huggingface/lerobot · GitHub"
[4]: https://huggingface.co/lerobot/smolvla_libero/blob/main/README.md?utm_source=chatgpt.com "README.md · lerobot/smolvla_libero at main"
[5]: https://github.com/huggingface/lerobot/blob/main/docs/source/env_processor.mdx?utm_source=chatgpt.com "lerobot/docs/source/env_processor.mdx at main · huggingface/lerobot · GitHub"
[6]: https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/libero.py "lerobot/src/lerobot/envs/libero.py at main · huggingface/lerobot · GitHub"
[7]: https://github.com/huggingface/lerobot/blob/main/docs/source/libero_plus.mdx?utm_source=chatgpt.com "lerobot/docs/source/libero_plus.mdx at main · huggingface/lerobot · GitHub"
[8]: https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan.pdf?utm_source=chatgpt.com "Using Codex with your ChatGPT plan | OpenAI Help Center"
