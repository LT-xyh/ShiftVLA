# M1-SA-v1：科学准入契约

**状态：APPROVED。2026-09-15 用户批准 M1-SA-v1 作为 `xyh/replayvla-p1` 的新 scientific-admission authority；动态工作仍须严格遵守 F0→F6 阶段权限与前置 gate。**

## 1. Authority 与迁移边界

基线为 `e37dfa93dfcc5793340829f8c685a4f5ec8c172b`。旧 design `a511e05034c3c65a0a034226c625b091e0b2e319`、旧 plan `16f27942825b078d1794a4c5f9610272f932d6cf`、原 G1、E2.3 canonical JSON、E2 closure、旧运行目录均不覆盖、不回填、不删除。

新路径命名为 M1-SA-v1；科学研究增补命名为 ReplayVLA-P1。新路径结果使用独立的 `SA-runtime`、`SA-renderer`、`SA-null`、`SA-physical-replay`、`SA-policy-replay` 状态。新 PASS 不等于旧 G1、旧 R1 Renderer Preflight 或旧 M1-N0 PASS。E3–E6/G2 不因新路线执行而被开启或补写完成。

新契约只改变第三方执行边界的保证方式。完整状态、静态物理模型、离散比较、null 校准、失败保留、固定动作和随机性等科学要求仍必须满足。（R1–R6）

旧 scientific spec 关于 Δ/R 的定义不改；ReplayVLA-P1 增加行为诊断的主问题并调整首篇论文工作顺序。把 activation exchange 延后到有预算时，是需要随本包明确批准的范围调整，不得以此声称原 ShiftVLA 全部里程碑完成。（R7）

## 2. 实验假设与可信计算边界

### 2.1 明示信任

在锁定的实验主机与权限域内，信任操作系统、CPython、NumPy、Torch、MuJoCo、LIBERO、robosuite、renderer/driver 的固定安装产物和约定实现。假设无恶意第三方代码、无未授权文件替换、无外部进程修改实验状态。这个假设必须出现在论文局限与运行 manifest 中。

固定版本号不够：记录实际解释器、模块来源、wheel/安装产物摘要、关键 DSO、资产、checkpoint、配置、renderer 身份和实验程序 commit。摘要只表明身份；未取得的 source-to-binary build provenance 标为 `NOT_INDEPENDENTLY_ATTESTED`，不能写成已证明。（W3）

不再以建立覆盖所有 native 内部执行、所有启动 hook、所有监控抑制区、所有未知 callback 的独立非执行证明，作为新路线每次实验的前提。也不实现通用攻击防御沙箱、完整 executable-payload 认证体系或 native fatal primitive。

### 2.2 自有 harness 仍须提供的保证

所有自有实验入口必须具有明确阶段、允许操作、计数边界和输出契约。测试覆盖重复调用、遗漏字段、漂移、异常、越阶段调用、失败记录和 no-overwrite。未知的科学相关可变 owner 仍使状态闭合失败，不能以“第三方可信”为由略过。

自有入口的错误必须使 attempt 终止为 FAIL/BLOCKED；父进程记录 return code、超时和清理结果。保护范围是声明的受控实验路径，不宣称能阻止任意恶意 native 代码或任意 Python catch 后的所有副作用。

正常 close 与进程结束应被直接检查。失败 cleanup 为 best effort；父进程只管理自身创建的子进程组。未覆盖的异常销毁行为应记录，而不是伪造安全退出。

### 2.3 Evidence 分类

| 类型 | 允许表达 | 不允许表达 |
|---|---|---|
| `OBSERVED` | 具体入口、具体时间和具体配置下直接测得的调用/状态 | 自动扩展到未运行路径 |
| `SOURCE_ACCOUNTED` | 某固定源码和明确条件支持的调用说明 | 冒充实测计数 |
| `TRUSTED_DEPENDENCY` | 固定第三方实现处于声明信任边界内 | 冒充 native absence proof |
| `NOT_OBSERVED` / `UNRESOLVED` | 当前没有足够证据 | 填成 `observed_count=0` |

## 3. 数据与验证/使用一致性

官方 task init-state 资产的窄用途反序列化，仅在新契约批准且相应阶段开放后允许；不授予任意 `torch.load`、checkpoint 或网络 payload 的通行权。需固定任务、容器、实际 byte hash、路径和读取条件。E2 generic VM 始终不保留物理 literals；原 E2 authority 不改。

真正读取的资产必须与认证对象一致：优先使用经核验且受控只读的资产快照、固定解析路径和受信 OS 下无并发替换的执行目录；能使用同一已验证字节/文件句柄时优先这样做。manifest 写出实际保障与假设，而不是只传 `asset_sha256` 常量给 extractor。

若 NumPy、Torch、资产或加载路径变化，原 E2 结论仅作为历史固定输入证据。必须核对受影响的语义与版本绑定；不能把旧 E2-PASS 当成新二进制 provenance 或 byte-use 证明。

动态 replay 阶段可在专用、非可执行 sidecar 中保存必要物理状态供比较。只使用无 pickle 的数值格式，验证 dtype/shape/hash。概述日志不打印原始 simulator 数组；generic symbolic VM 和 authority JSON 仍保持物理 opaque。这是新阶段的科学测量权限，不是放宽旧 E2 的解析器规则。

## 4. 冻结且不降低的科学门槛

### 4.1 完整状态与模型身份

保留 `mjSTATE_INTEGRATION` 的精确写回/readback；controller/OSC/interpolator、gripper、wrapper counters、observable timers/cache、task predicates、RNG、plugin/sleeping 状态均需明确归属。状态按 SERIALIZED / DERIVED_RECONSTRUCTED / IMMUTABLE / PROVEN_UNUSED 分类；未支持的可变 owner 使相关 gate BLOCKED。（R4、W1）

clean replay 的 physics 与 observation model fingerprints 均须相同。不得修补 physics-relevant mjModel 来掩盖恢复错误。每次 camera intervention 只能改冻结的 observation 字段，必须验证 physics identity 不变。

科学上的 exact-state 包含精确完整状态闭合和冻结的动态一致性检验；若浮点动态标准依赖独立 null envelope，论文必须如实写明，而非声称所有浮点轨迹普遍 bitwise identical。

### 4.2 构造与恢复严格分段

新 SA-renderer qualification 允许一次官方外层 construction reset，并记录其真实内部 init-state/reset/settle 路径；这些构造期行为不写成零。这与旧 R1 的 no-outer-reset preflight 不是同一个测试。

构造完成后开始纸面 action index 0。记录 simulator 原始 timestep 与实验步数的对应关系；不能把构造期 dummy/settle 当成研究动作。

一旦进入 restore，禁止 reset、set_init_state、settle、dummy action、autoreset、补动作对齐或物理模型修补。只有恢复字段、必要的受审 derived recomputation 和规定的后缀动作。h=0 读取不得通过强制刷新或额外 step 悄悄改变状态。（R4）

### 4.3 Null 与物理 replay

保留至少 20 对完整、独立 duplicate controls；每个所选 trace 至少 5 对、每个 regime 至少 5 对，且 contact/grasp/carried 必须有正证据。单次主 schedule 预先固定，不以失败后补跑凑足成功数。（R4–R6）

动作字节、contact identity、predicate、termination、counter、gripper 离散状态必须精确一致。浮点采用原冻结的逐组物理 oracle；不创建新全局 epsilon，不加事后系数，不因 RGB 差异扩大 physics tolerance。原 oracle 的其他精确要求同样保留。

每个已有七阶段窗口做 3 次同进程及 3 次 fresh-process restore。不能只做 contact/grasp/carried 三个窗口就声称完整 M1 通过。

每个新 task/model fingerprint、renderer、intervention 支持域和新的比较 horizon 都需要对应有效 controls。旧 task-0/null 不能自然外推到所有任务、camera 或 280-step policy 分支。

### 4.4 Policy replay 是另一道门槛

固定动作 tape 验证时 policy/processor 不运行。进入 policy 阶段后，使用官方 checkpoint 和 preprocessing，显式绑定噪声，恢复有效 policy history、action queue、cache、processor state 和所有相关 RNG。

不因 `n_action_steps=1` 就假设模型没有状态。旧 baseline 的 `action_noise.explicit=false` 不是 paired-noise 证据。（R8）

先通过 original vs restored 的完整无干预 policy 续跑和 sham switch controls，再运行真正的四分支。条件不同产生的新动作允许分叉；no-intervention 下不允许用宽松浮点误差掩盖行为分歧。

## 5. 阶段权限矩阵

下表自本文件所在 authority commit 起生效。阶段 PASS 为机械依赖，不等于每阶段要重新人工确认；不得提前使用后续阶段权限。

| 阶段 | 新权限 | 仍禁止 |
|---|---|---|
| F0：版本化契约与静态准备 | 只读核验、添加新计划/配置 schema、synthetic tests | 实验包导入以探行为、环境、EGL、policy、schedule |
| F1：runtime qualification | 精确解释器下所需 NumPy/Torch 导入；纯合成数组 roundtrip；枚举安装/模块/设备信息；至多两个隔离兼容性候选 | 环境、真实 init-state 解码、EGL context、policy 实例/权重加载、科学 rollout |
| F2：renderer qualification | 强审查后，官方资产加载、一次官方初始化、单个公共 render、close、三个 fresh worker 的结束证据 | policy/processor调用、研究 step、replay、null schedule |
| F3：null + physical replay | F2 PASS 后先登记新 null schedule；官方初始化和固定动作 tape；null PASS 后 capture/restore 与七阶段后缀 | policy、camera 干预、再生旧 tape、放宽 oracle |
| F4：policy-state/matched-noise/sham | F3 PASS 后固定 checkpoint、官方 processor、显式噪声和可复现 policy 续跑；关闭 RTC/异步；合格域的 camera sham | 训练、held-out 主结果、真实机器人 |
| F5：development pilot | F4 PASS 后 approved development roots、camera 处理、四分支、小型分析；新 task 必须先资格确认 | 读取 held-out 结果、在线调阈值、任意新 intervention |
| F6：冻结主实验与写作 | F5 判定可检验且协议 hash 冻结后，自动登记并执行一次 held-out schedule、分析与初稿 | 自适应加样本、成功后择优、训练、改主假设或预算 |

F1 允许维护“隔离的兼容性候选”不等于允许升级整套依赖。固定 MuJoCo/LeRobot/LIBERO/robosuite scientific behavior、checkpoint、动作/状态接口。若必须改变这些，停止并请求路线决策。允许的兼容性变更均要有问题定位、具体 delta、新 lock、重新 qualification；旧环境不被覆盖。

## 6. Gate、重试、cohort

阶段标准为 PASS / FAIL / BLOCKED / NOT RUN。代码测试失败与实证 attempt 失败分开：前者可修复后重测；后者单次 attempt 不重试、不覆盖。

工程原因需要新 qualification cohort 时，可在 10 日预算内最多替换一次：必须有具体代码/config/runtime delta、保留旧 cohort 的 FAIL/BLOCKED、新 ID、新完整 schedule。不能不改原因只反复抽一个成功 cohort。

未经测量的候选不是 PASS。主实验 schedule 冻结后不自动重跑失败行。技术缺失与任务失败不同：环境崩溃不能直接算策略成功，也不能静默删除；整体结果标为不完整并报告界限或缺失敏感性。

## 7. 投入与停止

D3 未定位具体运行时问题或没有合格候选，停止后续实验开发；D10 仍未完成物理及最小 policy replay 判断，停止扩大基础设施。有效 FAIL 同样是有价值的阶段结论，但不是进入主实验的资格。

暂停并升级的条件：科学相关状态不闭合、需要修改 oracle/任务/模型定义、无法获得真实 byte-use 绑定、需要超过批准的 runtime candidate 数量或投入预算、出现证据污染、需要新 native probe 或旧 E3/G2。

不得把暂停解释成静态证明无解、replay 不可能或首篇论文方向已经证伪。

## 8. 审批与自动推进

主代理负责最终判定；正常 substantive workers 默认 luna/max，sol/medium 只用于限定窄任务。模型路由是偏好而非已发生事实；不可伪造不可用 reviewer 的结论。

首次 F2 真实运行前强审查一次，检查实际新契约、入口代码、scientific oracle、synthetic negatives、源/安装身份、已声明 native trust boundary 和输出保护。审查新契约，不要求先完成旧 G1。若独立强 reviewer 不可用，记录 unavailable；未经主代理真实完成同等审查不得放行。后续重大范围变更再强审查，不逐 checkpoint 重审。

本契约及 ReplayVLA-P1 的范围增补已于 2026-09-15 获用户明确批准。允许各阶段仅在前置真实 PASS 后自动推进至 F6；触发停止条件时必须暂停并升级决策。
