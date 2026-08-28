# DCU-PREFLIGHT evidence

**Verdict: `DCU-PREFLIGHT = PARTIAL PASS`.**

The frozen single-card gates passed for both explicit `compare` and
`closed-loop`.  The two-worker concurrency gate is not closed: both workers
were started by the parent, but child 0 failed before reset/decision/step
because the CPU child inherited `CUDA_VISIBLE_DEVICES=0`, which conflicts
with the frozen `MUJOCO_EGL_DEVICE_ID=8`.  Child 1 was terminated by the
parent while its manifest was still `RUNNING`.  Consequently there is no
concurrency throughput result and the immutable DCU runtime lock remains
`PROPOSED_PENDING`.

## Scope and frozen invariants

This is an infrastructure/preflight record, not a scientific baseline.  No
baseline, baseline success rate, or comparative scientific claim is made.
The evaluated target is the frozen SmolVLA checkpoint on LIBERO Spatial task
0, with `seed=2027`, `init_state_id=0`, `horizon=280`, `n_envs=1`,
`obs_type=pixels_agent_pos`, RGB observations at 360x360, and
`n_action_steps=1`.

The explicit phases are isolated; `compare` does not call `env.step`, and no
phase automatically chains to another.  `compare` reconstructs the same
initial observation from task/seed/init state, uses the same explicit
`(1,50,32)` noise tensor on CPU and DCU, and sends the first action through
the same CPU official postprocessor and environment postprocessor.  It only
resets/renders and performs the feature/chunk/action comparison.  This is a
reconstructed same initial observation, not a claim of a historic tensor
bitwise hash.

`closed-loop` uses the CPU simulator/render and all official processors; the
DCU worker only performs remote `select_action`.  It uses the native worker
Torch RNG seeded with 2027, no explicit noise, the official one-action queue,
and one real 280-step-maximum episode.  `concurrency` is deliberately a
one-step child probe, not two full episodes: the parent launches two
independent CPU env/process/worker children, intended mapping physical 0 and
1 to logical `cuda:0`, and each child should reset once, decide once, step
once, then exit.

Training, backward, LoRA/PEFT, hooks, replay, perturbation, Plus, matched
sampling, AMP/precision changes, fused backends, DDP, and tensor/model
parallelism were not used.

## Provenance and commits

The execution code SHA was
`295a61acf829868af36effebfcc4dd77eecb4aa2` (`fix: enforce eager attention
for DCU inference`).  Relevant implementation/evidence commits are:

| Commit | Role |
| --- | --- |
| `91495af1e2a6c296ca70498ed2f86a6dd6606403` | M0 smoke harness and verified evidence |
| `8dc0525830943b568958b51d4cf02e4434d25a16` | Preserve Git porcelain status paths |
| `f063e95c3c5ba0dcd9b8d83127a0acb45ec70cf8` | Fail-closed DCU preflight harness |
| `4726b82df6795217328cd6a9b7607b91c028d2b7` | Preserve the frozen renderer after EGL reindex |
| `144dccc4e3db088b9b8d173f0ee97f5885b310c8` | Sanitize exported shell functions for workers |
| `295a61acf829868af36effebfcc4dd77eecb4aa2` | Standard eager-attention compatibility selection |

The expected and actual project SHA in the accepted single-card runs both
equal `295a61acf829868af36effebfcc4dd77eecb4aa2`.  The accepted input
config is
[`configs/m0/dcu_preflight.yaml`](/public/home/xuyinghao/workspace/vla/configs/m0/dcu_preflight.yaml),
SHA256
`80932fc1b406cd4fc7b4aaa595804efe438a4a60b7e0d8dd4b1fc0e4a9a494d3`;
the resolved config artifact SHA is
`34127142a75aa40ef35ff9f0868f5ce9ad3d3f9ea122a08b9398bf548d050c03`.

The exact model and asset inputs were:

| Input | Repository/revision | Local path | SHA256 |
| --- | --- | --- | --- |
| Checkpoint | `HuggingFaceVLA/smolvla_libero` / `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | `/public/home/xuyinghao/workspace/vla/external/artifacts/smolvla_libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |
| Base | `HuggingFaceTB/SmolVLM2-500M-Instruct` / `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | `/public/home/xuyinghao/workspace/vla/external/artifacts/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467` | `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3` |
| Assets | `lerobot/libero-assets` / `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` | `/public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` | manifest `1a94ebb8cc42614744d8dc9ebdad16f5461006f8806cc7d75869118013198a85` |

Pinned source revisions recorded by the preflight are LeRobot
`7e241bd630a3719a56157a497ce5d08f244784f1`, hf-LIBERO
`8561c60eea2fb93096146f240194649df73d8b1e`, robosuite
`fbee5844ff5632f5b5698e204ec5357ca50be0df`, and MuJoCo
`72cb2b210da666617924de709406d6aadbe60c71`.  The approved LeRobot patch is
[`runtime/patches/lerobot-v0.6.1-python311.patch`](/public/home/xuyinghao/workspace/vla/runtime/patches/lerobot-v0.6.1-python311.patch),
SHA256
`3ebaba1e8d305c93d0c758c9c9d3a575215ebe3513e486ae5503da444f8d573d`; the
Python 3.11 overlay wheel is
`/public/home/xuyinghao/tmp/shiftvla-lerobot-py311/wheels/lerobot-0.6.1-py3-none-any.whl`,
SHA256 `d03459fe530b556398bab41ad7de60b47c3354ffa7a5ce44db6249571fcdd2ff`.

The archived M0 reference used for the first-decision metadata gate is
[`run_manifest.json`](/public/home/xuyinghao/workspace/vla/runs/m0_smoke/20260825T075758Z_2_f664c2fe/run_manifest.json),
SHA256 `bf5c510f1e2c9fbdb0c27dca55f8682bcaaaad6e9bc8c3385aa788501eb91506`,
with decisions SHA256
`6d4c65d0fb8e3829971fb8b1023641172683c7fcf019f831823719ccd2299267`.

## Runtime, device, and renderer evidence

The CPU interpreter was
`/public/home/xuyinghao/tmp/shiftvla-libero/bin/python` (Python 3.12.6,
`torch 2.11.0+cpu`, `transformers 5.5.4`, NumPy 2.2.6, CUDA unavailable,
HIP unavailable).  Its lock
[`shiftvla-libero-runtime.txt`](/public/home/xuyinghao/workspace/vla/runtime/locks/shiftvla-libero-runtime.txt)
is SHA256
`921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc`; the
runtime recorded an exact 139-line `pip freeze --all` with SHA256
`a7463f6bd55400948b4ae9735e085b0c68de2d15aabf187b41f066977edcbba8`, and
`pip check` returned exactly `No broken requirements found.`

The DCU worker used
`/public/home/xuyinghao/tmp/shiftvla-libero-dcu/bin/python3.11` (Python
3.11.16, `torch 2.7.1`, HIP `6.3.25405`, CUDA version `null`) and reported
one logical `cuda:0` device named `K100_AI`.  Its immutable input lock
[`shiftvla-libero-dcu-runtime.txt`](/public/home/xuyinghao/workspace/vla/runtime/locks/shiftvla-libero-dcu-runtime.txt)
is SHA256
`c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a` and
remains `PROPOSED_PENDING_DCU_PREFLIGHT` / `NOT_CLOSED`; its exact 139-line
freeze SHA is
`aa980a6cae3c6e7aea4c80f0f8086184d05903aae1140374f87dd5b6b4b8601a`, and
its real-time `pip check` also returned exactly
`No broken requirements found.`  `hipcc --version` returned 0 with dcc
25.08.0-0, clang 17, and `/opt/dtk-25.04.2/llvm/bin`.  The lock records
driver `6.3.16-V1.1.0a`, DTK root/version `/opt/dtk-25.04.2` /
`DTK-25.04.2`, DTK ROCm/dev/utils `25.04.2` and libraries `25.04.1`, CUDA
compatibility `12.6.77`, and hipconfig HIP `6.3.25422`; the worker torch HIP
is `6.3.25405`, so both values are retained rather than conflated.  `hy-smi` returned 0;
card 0 was externally occupied while card 1 was idle for the accepted
single-card runs.  The host exposed eight DCUs; the approved preflight scope
selected physical devices 0 and 1.  The lock's two-device probe mapped both
selected cards to `K100_AI`, reported `torch.cuda.is_available()=true` and
device count 2 under `HIP_VISIBLE_DEVICES=0,1`, and completed finite FP32 and
BF16 matmul/add checks on both logical devices.  The accepted model/forward
evidence is only for physical card 1, mapped to logical `cuda:0`; the failed
concurrency run produced no model evidence for card 0.  The CPU and DCU
Python environments are intentionally different (3.12 versus 3.11), and
the overlay remains a recorded risk.

`LIBERO_CONFIG_PATH` was the directory
`/public/home/xuyinghao/tmp/shiftvla-libero-config`; its explicit config file
[`config.yaml`](/public/home/xuyinghao/tmp/shiftvla-libero-config/config.yaml)
has SHA256
`98d57e0d3d7b70bab5c66a110ecc333c737895e9dcd457c06ec94b4150e0ecc8`.
All accepted runs used fresh empty phase-specific HF/MPL/XDG cache
directories, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`HF_DATASETS_OFFLINE=1`, telemetry and implicit-token suppression,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, and the pinned pip cache.
The IPC boundary was tensor-only safetensors under each phase run directory;
no pickle or tensor-bearing JSON IPC was used.

The host EGL enumeration found 9 device ordinals.  Ordinals 0 through 7
failed because `/usr/lib64/dri/hycu_dri.so` was unavailable; ordinal 8
passed.  The frozen renderer identity was preserved exactly, not changed:
vendor `Mesa/X.org`, renderer `llvmpipe (LLVM 12.0.0, 256 bits)`, OpenGL
version `3.1 Mesa 21.1.5`.  The accepted config therefore uses
`MUJOCO_EGL_DEVICE_ID=8`; this is a host-specific ordinal correction that
preserves the exact renderer identity.  `compare` reset/render GL evidence,
`closed-loop` episode trace GL evidence, and the one-step child trace schema
all record this identity where the child reached the trace seam.

The retained preflight evidence indexes are:

- [`compare/preflight.json`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T073511Z/preflight.json)
  (SHA256 `281a5a51022cf42a7a13a7ffd02c350ac030ddac4d7ef8c9c813b4b35026ab48`)
- [`closed-loop/preflight.json`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T073911Z/preflight.json)
- [`concurrency/run_manifest.json`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T074455Z/run_manifest.json)

Each preflight includes the exact lock paths/SHA, freeze/check result,
Python executable, `hy-smi` and `hipcc` command/output/return code, offline
variables, and renderer variables.  Each accepted result also records the
actual GL vendor/renderer/version and the worker runtime/hardware evidence.

## Accepted `compare`

Run directory:
[`20260827T073511Z`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T073511Z).
The command exited 0 with status `PASS`, elapsed 90.4097585548 s.  Key
artifact hashes are: `comparison.json`
`41f931e69367ccf086ef7724cdb373356bce42c227adc4767ba60849a7ffcaf1`,
`decisions.jsonl`
`184b397d168b9f3d32bf13e93e0916e889eaf7d62490021b3c3e51d2b928c1c2`,
`command.txt`
`e32f33c1b72af2eb243af945e26346b75fa4d75d89deaa0074973c6d50dc1a37`,
request safetensors
`ddc0287f2be70c85d9e892b82af9093f502c3b0949d3edad5ac33e1402d0c1c8`, CPU
chunk safetensors
`d72a121211635a7348b137e5bf126e66a854e0b1b1365dc3c7436403252f1f22`, and
DCU chunk safetensors
`1daa598021df3fdd3423cc938d6af14f871cf6c35671d7e40db742d3509457d0`.

The explicit noise seed is 2027 and noise SHA256 is
`0919ab411bdcac49d637839a4cd16e105184efcd39fb7e2a3b95967736dd06b3`.
Every required feature was finite and matched the archived M0 first-decision
metadata: `image` and `image2` are `[1,3,360,360]` float32, `state` is
`[1,8]` float32, language tokens are `[1,20]` int64, and the attention mask
is `[1,20]` bool.  Both model chunks are `[1,50,7]` float32 and the selected
actions are `[1,7]` float32.  Reference floating min/max values were checked
with the recorded absolute/relative tolerance `1e-7`; the comparison says
`same_initial_observation` and does not claim historical tensor-bitwise
identity.

The normalized action-chunk difference was max
`0.08260536193847656`, mean `0.013760031765060765`; final-action difference
was max `0.017833083868026733`, mean `0.005867380382759231`.  Both chunks
and actions were finite.  CPU wall/inference latency was
29.3708877095 s; DCU worker inference was 2.3399509881 s and worker wall
latency was 2.4051448833 s; render latency was 0.0001464710 s.  The model
load latency was 11.3018531445 s, model peak memory was 2,483,390,464 bytes,
and response peak was 1,423,232,512 bytes.  `env_step_count=0`.

The worker loaded `SmolVLAPolicy` strictly (`strict_checkpoint_load=true`),
with 604,934,176 parameters, 746 bfloat16 and 42 float32 parameter tensors,
all on logical `cuda:0`; training, AMP, PEFT, and compile were false,
`chunk_size=50`, `n_action_steps=1`.  Startup bf16 and fp32 tensor probes
were `[2,2]`, finite, on `cuda:0`.  The requested attention backend was
eager; effective attention was eager for action expert, VLM, VLM text, and
VLM vision, while the prior setting was SDPA.  No forbidden-backend error
occurred and the worker-side gate completed; this is evidence for the
approved eager backend selection, not a claim that the runtime is a
baseline.

The full accepted command was:

```bash
env HF_HOME=/public/home/xuyinghao/tmp/dcu-preflight-compare-hf-Ue4wga MPLCONFIGDIR=/public/home/xuyinghao/tmp/dcu-preflight-compare-mpl-rfEf5O XDG_CACHE_HOME=/public/home/xuyinghao/tmp/dcu-preflight-compare-xdg-7H6IKA HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 LIBERO_CONFIG_PATH=/public/home/xuyinghao/tmp/shiftvla-libero-config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_CACHE_DIR=/public/home/xuyinghao/tmp/shiftvla-pip-cache /public/home/xuyinghao/tmp/shiftvla-libero/bin/python /public/home/xuyinghao/workspace/vla/scripts/dcu_preflight.py compare --config /public/home/xuyinghao/workspace/vla/configs/m0/dcu_preflight.yaml --expected-project-sha 295a61acf829868af36effebfcc4dd77eecb4aa2 --physical-device 1
```

## Accepted `closed-loop`

Run directory:
[`20260827T073911Z`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T073911Z).
The command exited 0 with status `PASS`, elapsed 140.4819853660 s.  The
episode result SHA256 is
`af00ca94c49454d1e4ee5c990e67bd4c2fe0d6f86fb25a68204ed3e52102f039`,
decisions SHA256 is
`cee1789cc8b8ee66b6810260d51bd1d0ebcdc2733208c0b96398d4cba72d1d95`,
preflight SHA256 is
`0d39faed448e33951f63e0586af951bff8a591c72e5f7ab0e6755e70bd09031f`, and
command SHA256 is
`0bd51adb032909040d33ca678ce05d8006b609afc3c149775ebd88487fd417e0`.

Task evidence was checked through the pinned task/BDD/init seams without an
extra reset: suite `libero_spatial`, task 0,
`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`,
description “pick up the black bowl between the plate and the ramekin and
place it on the plate”, `init_state_id=0`, and horizon 280.  The BDDL file is
[`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl`](/public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl);
the init-state file is
[`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init`](/public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init).

There was exactly one reset, 82 policy calls, 82 decisions, 82 real
environment steps, and 83 renders.  The official done flag became true on
the final step; `steps=82`, `completed_by=done_or_horizon`,
`termination_reason=environment_termination`, `success_observed=true`, and
the reward sum/final reward were 1.  The success threshold was not enforced.
Every remote action, normalized action, official postprocessed action,
environment-postprocessed action, and env-step action was `[1,7]` float32
and finite.  `env_step_equals_postprocessed=true` for 82/82 decisions.
The official queue was retained with `n_action_steps=1`: queue length before
and after was 0 and a new chunk was generated for all 82 decisions; worker
queue evidence agrees.  The final trace records `done=true`, reward 1, and
success true.

Using nearest-rank p95 over the 82 decision records, the latency summary is:

| Measurement | Mean (s) | p95 (s) | Max (s) |
| --- | ---: | ---: | ---: |
| Policy/remote inference | 0.6470162260 | 0.6901538484 | 2.3058810011 |
| Worker inference | 0.5576647263 | 0.5646023452 | 2.1881470922 |
| Worker wall | 0.6024428015 | 0.6259194445 | 2.2605833355 |
| Environment step | 0.2813922646 | 0.2949349880 | 0.3148635961 |
| Render (83 samples) | 0.0002249969 | 0.0003288332 | 0.0003993250 |

Throughput is reported only at the seams actually measured.  The one-sample
`compare` latency corresponds to a CPU reference proxy of 0.03405 decisions/s
and a one-K100 worker-wall proxy of 0.41578 decisions/s.  Across the 82-step
closed loop, reciprocal mean worker-wall latency is 1.65991 decisions/s; this
includes the first-call warm-up in the mean and is not a baseline throughput
benchmark.  Two-worker throughput is unavailable because neither child pair
completed the concurrency gate.

Worker load latency was 11.9281353932 s; model peak memory was
2,483,390,464 bytes and per-select response peak was 1,423,232,512 bytes.
The model/device/dtype/eager/basic-operation evidence matches compare:
strict SmolVLAPolicy load, logical `cuda:0` / `K100_AI`, torch 2.7.1 / HIP
6.3.25405, 746 bfloat16 and 42 float32 parameter tensors, no training/AMP/
PEFT/compile, finite `[2,2]` bf16/fp32 startup probes, prior SDPA and
effective eager attention.  The episode trace records the exact GL identity
above.  Closed-loop action noise is explicitly documented as
`native_worker_torch_rng`, seed 2027, not an explicit same-noise hash.

The full accepted command was:

```bash
env HF_HOME=/public/home/xuyinghao/tmp/dcu-preflight-closed-hf-WfKI1Y MPLCONFIGDIR=/public/home/xuyinghao/tmp/dcu-preflight-closed-mpl-bPDVbC XDG_CACHE_HOME=/public/home/xuyinghao/tmp/dcu-preflight-closed-xdg-7twVBJ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 LIBERO_CONFIG_PATH=/public/home/xuyinghao/tmp/shiftvla-libero-config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_CACHE_DIR=/public/home/xuyinghao/tmp/shiftvla-pip-cache /public/home/xuyinghao/tmp/shiftvla-libero/bin/python /public/home/xuyinghao/workspace/vla/scripts/dcu_preflight.py closed-loop --config /public/home/xuyinghao/workspace/vla/configs/m0/dcu_preflight.yaml --expected-project-sha 295a61acf829868af36effebfcc4dd77eecb4aa2 --physical-device 1
```

## Failed `concurrency`

Run directory:
[`20260827T074455Z`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T074455Z).
The parent exited 1 after 21.0959204845 s with
`concurrency child 0 returned 1`.  It generated two explicit child commands:

```text
one-step-child --config /public/home/xuyinghao/workspace/vla/configs/m0/dcu_preflight.yaml --expected-project-sha 295a61acf829868af36effebfcc4dd77eecb4aa2 --physical-device 0 --run-directory /public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T074455Z/children/child0 --internal-token shiftvla-internal-one-step-v1
one-step-child --config /public/home/xuyinghao/workspace/vla/configs/m0/dcu_preflight.yaml --expected-project-sha 295a61acf829868af36effebfcc4dd77eecb4aa2 --physical-device 1 --run-directory /public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T074455Z/children/child1 --internal-token shiftvla-internal-one-step-v1
```

The full parent command was:

```bash
env HF_HOME=/public/home/xuyinghao/tmp/dcu-preflight-concurrency-hf-KZGqVj MPLCONFIGDIR=/public/home/xuyinghao/tmp/dcu-preflight-concurrency-mpl-NJYzmy XDG_CACHE_HOME=/public/home/xuyinghao/tmp/dcu-preflight-concurrency-xdg-JxXOtm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 LIBERO_CONFIG_PATH=/public/home/xuyinghao/tmp/shiftvla-libero-config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_CACHE_DIR=/public/home/xuyinghao/tmp/shiftvla-pip-cache /public/home/xuyinghao/tmp/shiftvla-libero/bin/python /public/home/xuyinghao/workspace/vla/scripts/dcu_preflight.py concurrency --config /public/home/xuyinghao/workspace/vla/configs/m0/dcu_preflight.yaml --expected-project-sha 295a61acf829868af36effebfcc4dd77eecb4aa2
```

Child 0's
[`failure_manifest.json`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T074455Z/children/child0/failure_manifest.json)
records the exact pre-reset error:
`pinned CPU runtime imports failed: MUJOCO_EGL_DEVICE_ID needs to be set to
one of the device id specified in CUDA_VISIBLE_DEVICES`.  Thus child 0 has
reset=0, decision=0, env.step=0, and no worker/model/GL/action/queue
evidence.  Child 1 was still `RUNNING` in its
[`run_manifest.json`](/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260827T074455Z/children/child1/run_manifest.json)
when the parent terminated it; it has no valid child result.  There is no
`concurrency.json`, no completed child pair, and no throughput value to
report.  No process was left running and the externally occupied card 0 was
not killed or disturbed.

This is an orchestrator environment-propagation bug, not evidence that a
two-worker model or device computation completed.  The frozen concurrency
mapping and one-step contract remain unaccepted until children receive a
consistent CPU EGL environment and both produce complete result manifests.

## Diagnostic timeline before the accepted runs

These no-overwrite diagnostics explain the progression and are not silently
substituted for acceptance:

| Run | Code/config condition | Result |
| --- | --- | --- |
| `20260827T070651Z` | `f063e95c...`, renderer ordinal 0 | CPU LIBERO construction failed with EGL device-display/`PLATFORM_DEVICE` ImportError; `env_step_count=0`. |
| `20260827T071630Z` | `4726b82d...`, renderer ordinal 8 | CPU reset/render and feature preparation succeeded; worker environment rejected an invalid NUL/newline before worker execution. |
| `20260827T072140Z` | `144dccc4...`, renderer ordinal 8 | CPU setup and worker start/reset succeeded; prediction failed because no `flash_attn_2_cuda*.so` matched. |
| `20260827T073511Z` | `295a61a...`, eager backend | `compare` PASS. |
| `20260827T073911Z` | `295a61a...`, eager backend | `closed-loop` PASS. |
| `20260827T074455Z` | `295a61a...`, two children | `concurrency` FAIL before any child step. |

## Gate summary

The labels below preserve the acceptance criteria from the approved
DCU-PREFLIGHT request.  Two-worker concurrency is an additional conditional
stage, not a relabeling of criteria A-G.

| Gate | Evidence | Status |
| --- | --- | --- |
| A. K100 visible to PyTorch | The two-device lock probe identifies physical 0/1 as `K100_AI`; the accepted worker positively identifies card 1 and finite FP32/BF16 operations. | PASS |
| B. Exact frozen SmolVLA loads | Local-only checkpoint/base artifacts load strictly; 604,934,176 parameters and all parameter devices/dtypes are recorded, with no checkpoint or architecture change. | PASS |
| C. Real observation produces finite 7D action | Official processing of the reconstructed task-0 observation produces finite `[1,50,7]` chunks and finite `[1,7]` postprocessed actions. | PASS |
| D. CPU/DCU deviation quantified | Same explicit noise yields the recorded nonzero max/mean chunk and final-action differences; neither bitwise nor numerical identity is claimed. | PASS |
| E. One closed-loop DCU episode executes real steps | Exactly one episode executes 82 policy calls and 82 real `env.step` calls, then terminates with observed success/reward 1. | PASS |
| F. No unsupported fused dependency introduced | Effective attention is standard eager; the worker's runtime guard completed without observing its forbidden module set.  The guard-coverage mismatch below remains a fail-closed risk. | PASS WITH RESIDUAL RISK |
| G. Provenance/runtime lock recorded | Source/artifact/config/runtime hashes, commands, device/toolchain/GL evidence and no-overwrite run directories are recorded.  The lock remains proposed and the provenance hardening gaps below remain open. | PASS WITH RESIDUAL RISK |

The conditional independent two-worker stage is **FAIL / OPEN**: child 0
failed before reset because inherited `CUDA_VISIBLE_DEVICES=0` conflicts with
EGL ordinal 8, and child 1 remained stale `RUNNING`.  Therefore the overall
preflight is `PARTIAL PASS` exactly as required when one-card inference and
closed loop pass but two-worker concurrency does not.

## Open risks and acceptance boundary

- The concurrency orchestrator must isolate CPU-child CUDA/EGL environment
  variables; the current inherited `CUDA_VISIBLE_DEVICES=0` conflicts with
  `MUJOCO_EGL_DEVICE_ID=8`.
- Child 1's stale `RUNNING` manifest must not be treated as a result.
- Card 0 remains externally loaded; single-card accepted phases were pinned
  to physical card 1 for that reason.
- The DCU worker emitted a NumPy ABI warning: a module compiled against
  NumPy 1.x was loaded with NumPy 2.2.6 and `_ARRAY_API` initialization was
  unavailable.  Tensor-only safetensors probes and actions passed, but the
  ABI warning remains unresolved.
- CPU and DCU Python/torch environments differ (CPU 3.12.6/torch 2.11.0+cpu
  versus DCU 3.11.16/torch 2.7.1), and the Python 3.11 overlay remains part
  of the provenance rather than an unqualified equivalence claim.
- Eager attention is the approved, explicitly recorded backend selection
  after the earlier missing-flash-attention diagnostic; it is not a claim
  of an unmodified flash/fused backend or a baseline.
- The command-line `--expected-project-sha` is checked against current HEAD,
  but the caller supplies that expected value.  The successful manifests
  prove agreement with the recorded invocation, not an independently
  immutable project-code pin.  A later trusted launcher/config seam should
  freeze the accepted SHA rather than accepting it solely from the caller.
- The worker guard rejects the observed/expected module names including
  `flash_attn`, Mamba, Triton, xFormers, and bitsandbytes, but its code list
  does not include the literal alias `flashattention` that appears in the
  runtime lock's broader inventory.  No such distribution, module spec, or
  loaded module was observed, and effective attention was eager, but the
  list mismatch remains a fail-closed coverage gap.
- Direct invocation of `dcu_model_worker.py` does not independently enforce
  every parent-side artifact, runtime-lock, offline, task, or project-SHA
  gate.  The accepted evidence uses the guarded parent path; the worker
  should later require a trusted parent contract before it is reused.
- The run manifests record task, seed, init state, explicit/native action
  noise mode, and `scientific_scope.perturbation=false`, but do not expose
  dedicated `trajectory_id` and structured `perturbation_settings` fields.
  Those fields remain required before scientific experiment manifests are
  accepted; this preflight is infrastructure evidence only.
- The immutable input DCU lock
  [`shiftvla-libero-dcu-runtime.txt`](/public/home/xuyinghao/workspace/vla/runtime/locks/shiftvla-libero-dcu-runtime.txt),
  SHA256 `c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a`,
  remains `PROPOSED_PENDING` because the full two-worker Gate G is not
  closed.  This document does not modify that lock.

No training, baseline, or scientific M0 result should be inferred from this
partial infrastructure verdict.
