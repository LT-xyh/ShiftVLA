# ShiftVLA M0-PREFLIGHT

Date: 2026-08-24

ShiftVLA bootstrap HEAD at preflight start:
`1c9cf2b4a2e441183c750818e3d21669f7f39358`

Initial specification anchor:
`53ab983e6fa32f26d5d223927810be022ff5bc6f`

## Current phase gate (2026-08-25)

- **M0-PREFLIGHT-A:** `PASS` (human-approved; the earlier A evidence is
  retained below as historical evidence).
- **M0-PREFLIGHT-B:** `PASS` (main-reviewed). Artifact materialization,
  vanilla reset/render, strict local-only load, and one real
  observation-to-finite-action forward gates passed.
- **M0:** `NOT_STARTED`; this phase does not automatically authorize M0.
  Wait for the next explicit instruction before any baseline rollout or
  experiment work.
- Mesa `llvmpipe` and the absence of visible HCU hardware remain unresolved
  performance/runtime risks. No HCU acceptance is claimed.

## Historical M0-PREFLIGHT-A scope and outcome

This phase did not run a baseline episode and did not implement M0, replay,
perturbations, hooks, activation patching, LoRA, training, or any scientific
runtime code. Existing entries and SHAs in `external/pins.yaml` were not
changed.

The historical Preflight-A outcome was **BLOCKED, not M0-ready**; this section
is retained for provenance and is not the current B result:

- **Checkpoint recommendation:** retain the currently pinned
  `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
  Do not switch to `lerobot/smolvla_libero` silently.
- **CPU software lock:** resolved and import-checked in the isolated
  `/public/home/xuyinghao/tmp/shiftvla-libero` Python 3.12 environment.
  `pip check` passed. This is a CPU-only preflight result, not an HCU lock.
- **EGL:** MuJoCo 3.7.0 rendered through EGL, but OpenGL identified Mesa
  `llvmpipe`; no HCU device nodes were visible. HCU EGL therefore failed.
- **Vanilla LIBERO:** the pinned package, suite, task, BDDL, and initial state
  imported successfully. Reset and RGB rendering were not attempted after the
  fail-closed asset gate found that exact-revision assets were not
  materialized.
- **Model:** the selected SmolVLA config and the exact local base-model
  processor/tokenizer metadata loaded with an empty Hub cache and offline
  mode. The serialized normalizer's Git-LFS pointer failed in the safetensors
  parser without a Hub fallback; this is not a substitute for the proposed
  explicit hash gate. Checkpoint/base weights could not be materialized, so
  full model loading and the one-action forward were not run.

The asset download was requested at its exact revision. The execution platform
rejected the network escalation because its usage quota was exhausted. Model
and base-model downloads were not attempted after that gate failed. No mirror,
mutable branch, cached substitute, random model, or other workaround was used.

The sections from **Evidence sources** through **Unresolved risks and approval
gates** below are retained historical M0-PREFLIGHT-A evidence. Their earlier
asset/model `BLOCKED` and `NOT RUN` statements describe the A snapshot; the
current B result is recorded in the addendum at the end of this document.

## Evidence sources

### Pinned source

| Component | Absolute path | Revision |
| --- | --- | --- |
| LeRobot v0.6.1 | `/public/home/xuyinghao/workspace/vla/external/lerobot` | `7e241bd630a3719a56157a497ce5d08f244784f1` |
| HF vanilla LIBERO 0.1.4 | `/public/home/xuyinghao/workspace/vla/external/hf-libero` | `8561c60eea2fb93096146f240194649df73d8b1e` |
| robosuite 1.4.0 | `/public/home/xuyinghao/workspace/vla/external/robosuite` | `fbee5844ff5632f5b5698e204ec5357ca50be0df` |
| MuJoCo 3.7.0 source | `/public/home/xuyinghao/workspace/vla/external/mujoco` | `72cb2b210da666617924de709406d6aadbe60c71` |
| selected checkpoint metadata | `/public/home/xuyinghao/workspace/vla/external/models/smolvla-libero` | `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` |
| selected base VLM metadata | `/public/home/xuyinghao/workspace/vla/external/models/smolvlm2-500m-instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

The three source-built Python packages (LeRobot, hf-libero and robosuite) were
built from the local pinned trees. Their installed `direct_url.json` records
point to those absolute paths. MuJoCo 3.7.0 came from its binary Python wheel,
not a build of the pinned C/C++ checkout; this distinction remains part of the
lock-strength risk below. All external worktrees were clean after moving
pip-generated, untracked build directories to
`/public/home/xuyinghao/tmp/shiftvla-m0-generated-build-artifacts`.

### Read-only upstream comparison metadata

The current `lerobot/smolvla_libero` branch tip was resolved, not treated as a
pin, and checked out with Git-LFS smudging disabled at:

```text
/public/home/xuyinghao/tmp/shiftvla-m0-upstream-metadata/lerobot-smolvla-libero
31d453f7edd78c839a8bbc39744a292686daf0de
```

The checkout was detached and clean. This comparison does not modify
`external/pins.yaml`.

## Checkpoint-family comparison

| Property | Current ShiftVLA selection: `HuggingFaceVLA/smolvla_libero@6721902…` | Upstream candidate: `lerobot/smolvla_libero@31d453f…` |
| --- | --- | --- |
| Artifact generation | 2025 checkpoint migrated to the current `PolicyProcessorPipeline` JSON format | 2026 current LeRobot-org generation, including `train_config.json` |
| v0.6.1 config parsing | **Runtime metadata probe passed** using local path and `local_files_only=True`; omitted newer fields receive v0.6.1 defaults | **Runtime metadata probe passed** using local path and `local_files_only=True` |
| State contract | `observation.state: (8,)` | `observation.state: (6,)` |
| Camera contract | `observation.images.image`, `image2`; two cameras | `camera1`, `camera2`, `camera3`; three declared cameras; processor renames only `image` and `image2` |
| Action contract | `(7,)` | `(7,)` |
| Chunk / queued execution | `chunk_size=50`, `n_action_steps=1` | `chunk_size=50`, `n_action_steps=50` |
| Named base VLM | `HuggingFaceTB/SmolVLM2-500M-Instruct` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| Serialized tokenizer | right side, `padding="longest"`, maximum 48 | right side, `padding="max_length"`, maximum 48 |
| Serialized processor | empty rename map; device; state/image normalization; CPU then action unnormalization | two-camera rename to `camera1/2`; device; state/image normalization; action unnormalization then CPU |
| SmolVLA depth/width | 32 VLM layers and 32 expert layers; expert width multiplier 0.5 | 16 VLM layers and 16 expert layers; expert width multiplier 0.75 |
| Training flags in config | frozen vision encoder, expert-only training flags | full-model flags (`freeze_vision_encoder=false`, `train_expert_only=false`) |
| Weight LFS object | SHA256 `71d9563c…ed52f`, 1,218,047,032 bytes | SHA256 `9a9f6413…afca8`, 906,712,520 bytes |
| Evaluation provenance | Model card is generic and says dataset unknown; no serialized train config. Its tensor contract matches the pinned official vanilla LIBERO processor and two-camera environment. | `train_config.json` says dataset `lerobot/libero`, environment `libero_spatial`, 360×360 and two cameras, but the policy inside the same file declares 6-D state and three cameras. Dataset revision is null. |
| Expected official evaluation path | `lerobot-eval` → `lerobot_eval.eval_main` → `make_env` / `make_policy` / `make_pre_post_processors` / `make_env_pre_post_processors` → `eval_policy_all` → `eval_policy` → `rollout` | Nominally the same path, but blocked before valid policy input by the pinned 8-D environment state versus 6-D checkpoint contract unless a prohibited adapter changes the definition |
| Native pinned runtime compatibility | **Best candidate.** The pinned LIBERO processor produces exactly 8-D state and the default environment maps exactly two cameras to `image/image2`. | **Not native.** The pinned processor produces 8-D state. Reaching 6-D would require a state-definition change or compatibility adapter. The declared third camera is absent, although the current image path can tolerate that absence. |
| Later activation intervention | Full 32-layer VLM/expert stacks preserve the architecture already audited for eight-group intervention | Still structurally instrumentable, but different depth/width and incompatible observation contract add avoidable confounds |

At the schema level, the selected 2025 config predates serialized
`use_peft`, `pretrained_path`, and `rtc_config` fields; pinned v0.6.1 supplies
their defaults when loading. The candidate explicitly serializes those newer
fields. Neither schema-load probe instantiated weights.

Selected evidence is in
`external/models/smolvla-libero/config.json:1-87`,
`policy_preprocessor.json:1-79`, and
`policy_postprocessor.json:1-32`. Candidate evidence is in the read-only
comparison checkout's `config.json:1-99`,
`policy_preprocessor.json:1-90`, `policy_postprocessor.json:1-32`, and
`train_config.json:1-231`.

Pinned LeRobot's decisive contract is source-defined:

- `external/lerobot/src/lerobot/processor/env_processor.py:26-105` creates
  `[eef position (3), axis-angle (3), gripper qpos (2)]`, hence 8-D state.
- `external/lerobot/src/lerobot/envs/configs.py:320-353` declares two cameras
  and maps them to `observation.images.image` and `image2`.
- `external/lerobot/src/lerobot/scripts/lerobot_eval.py:740-822` implements the
  evaluation entry and its factory/rollout handoff.

The candidate's missing third image is not by itself a hard failure:
`SmolVLAPolicy.prepare_images` processes present camera keys and only creates
configured empty cameras when `empty_cameras > 0`
(`modeling_smolvla.py:334-380`). The 6-D normalizer/policy state versus the
source-defined 8-D environment state remains incompatible.

### Recommendation

Retain the existing checkpoint pin for the ShiftVLA pilot, subject to passing
the still-blocked materialized model/forward probes. It satisfies the stated
priorities as follows:

1. It is the only family with a native observation contract matching pinned
   LeRobot v0.6.1.
2. It can use the official LeRobot vanilla LIBERO factory and processor path
   without truncating state, synthesizing a camera, or changing statistics.
3. The only required compatibility action is fail-closed local artifact
   routing; no mathematical or preprocessing patch is proposed.
4. Its complete 32-layer VLM and expert stacks are suitable for the later
   architecture-aware intervention design.

This recommendation does not claim recovered training provenance. The
checkpoint does not record its historical LeRobot, LIBERO, dataset, or base
VLM revision.

No checkpoint change is proposed, so no checkpoint entry should be added to
`external/pins.yaml`. If review instead selects the other family, add at least:

```text
lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de
HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467
```

The two SmolVLM repository names resolved to the same commit during this
read-only retrieval, but their identity and historical provenance must not be
treated as interchangeable. The candidate model card also names
`lerobot/smolvla_base`; pinning its then-current
`c83c3163b8ca9b7e67c509fffd9121e66cb96205` would be required for complete
provenance before any switch.

## Proposed vanilla runtime lock

Environment name/path used for this preflight:

```text
shiftvla-libero
/public/home/xuyinghao/tmp/shiftvla-libero
```

The complete observed resolution is recorded in
`runtime/locks/shiftvla-libero-cpu-preflight.txt`. Key versions are:

| Package | Resolved version |
| --- | --- |
| Python | 3.12.6 |
| pip | 24.2 |
| NumPy | 2.2.6 |
| PyTorch | 2.11.0+cpu |
| torchvision | 0.26.0+cpu |
| torchcodec | 0.11.1+cpu |
| Transformers | 5.5.4 |
| Hugging Face Hub | 1.26.0 |
| Gymnasium | 1.3.0 |
| LeRobot | 0.6.1, built from pinned local source |
| hf-libero | 0.1.4, built from pinned local source |
| robosuite | 1.4.0, built from pinned local source |
| MuJoCo | 3.7.0 |
| OpenCV packages | `opencv-python==4.14.0.94`; `opencv-python-headless==4.13.0.92` |

This resolution satisfies LeRobot's declared Python, torch, NumPy,
Transformers and Gymnasium bounds and hf-libero's robosuite/MuJoCo bounds.
`pip check` returned `No broken requirements found.` The active `cv2` reports
4.14.0.

After stripping comments and blank lines, the tracked lock is byte-for-byte
equal to the 139-line `pip freeze --all` output. The raw freeze SHA256 is
`a7463f6bd55400948b4ae9735e085b0c68de2d15aabf187b41f066977edcbba8`.
Because its three local-source requirements contain absolute `file://` paths,
this file is a host-specific resolution snapshot, not a portable accepted
runtime lock.

The lock is deliberately marked CPU-only. The host contains DTK 25.04.2
(`hipcc` reports 25.08.0-0), but its local PyTorch 2.7.1 wheels are only CPython
3.10 and 3.11 builds. No CPython 3.12 vendor wheel was found. In the current
execution context the CPU wheel reports `torch.version.cuda=None`,
`torch.version.hip=None`, and `torch.cuda.is_available()==False`;
`/dev/dri`, `/dev/kfd`, and `/dev/mkfd` are absent and `hy-smi` reports no
device. Therefore an accepted Python 3.12 HCU lock remains unresolved; the CPU
lock must not be promoted to M0.

This does not overwrite the earlier bootstrap observation in
`external/pins.yaml`, where a different host/device-visible context reported
eight `K100_AI` devices. That snapshot already marked every runtime acceptance
gate false. The current restricted preflight context exposes no device nodes;
the two observations are distinct, and neither is accepted HCU runtime
evidence for M0.

The current host's Python 3.13 / NumPy 2.4.6 environment was not used.

LIBERO-Plus must later use a separate environment, the pinned Plus fork, and
robosuite 1.4.1. It was not installed or imported here because all three
benchmark repositories expose the same `libero` namespace.

## Fail-closed artifact materialization design

### Required selected-checkpoint objects

| Artifact | Expected SHA256 | Expected bytes |
| --- | --- | ---: |
| checkpoint `model.safetensors` | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` | 1,218,047,032 |
| checkpoint preprocessor normalizer state | `7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e` | 416 |
| checkpoint postprocessor unnormalizer state | `7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e` | 416 |
| base VLM `model.safetensors` | `b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3` | 2,029,990,624 |

Checkpoint metadata hashes already recorded in `external/pins.yaml` are:

```text
config.json                e346a63094732ea28e05b08302a218306cf6541960b663a3cb4b6948e4efe6e0
policy_preprocessor.json    29267ff4423fcce93bbd76ad0f386905f9de5d267ca13379a5795a514a81b7a9
policy_postprocessor.json   ad2d22405cb63be4932c4a4d9a69bc9953df02a01235c5f420b679dbe6bfe7c5
```

The complete local base processor/tokenizer set required for local loading is:

| File | SHA256 |
| --- | --- |
| `added_tokens.json` | `74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a` |
| `chat_template.json` | `b585e3598909a5687f9f9d738d35223724dedef256b9b274e1cbfb32b13c74bf` |
| `config.json` | `ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17` |
| `generation_config.json` | `34835060c9f0f74d1acb456cc72ca32746d3843d9eb5f578f9cbffac1d2eb840` |
| `merges.txt` | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| `preprocessor_config.json` | `149e315d9410368e5491455bb06e0f763426e9e56cca731c13b24404a29b6374` |
| `processor_config.json` | `f3ad45028447b3562b4752be0d5916d6806c1ef589091a469608dcf0faa1737c` |
| `special_tokens_map.json` | `2dfea2a426162316ff1567c82bc6d36d9690cd9f90455f075c77daca78b45c60` |
| `tokenizer_config.json` | `dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913` |
| `tokenizer.json` | `5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a` |
| `vocab.json` | `82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79` |

### Vanilla LIBERO asset pin proposal

Pinned hf-libero 0.1.4 names the official
`lerobot/libero-assets` dataset but calls `snapshot_download` without a
revision (`external/hf-libero/libero/libero/utils/download_utils.py:109-159`).
Read-only upstream resolution found:

```text
https://huggingface.co/datasets/lerobot/libero-assets.git
refs/heads/main -> 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
retrieved 2026-08-24
```

The exact tree contains 586 files in six top-level asset directories,
177 LFS files, 143 unique LFS objects, and 422,320,936 declared bytes. Its
LFS pointer paths/OIDs/sizes matched the exact-revision metadata checkout.
This is a proposed additional runtime artifact pin; it was not added to
`external/pins.yaml`.

`get_assets_path()` first checks package-local `libero/libero/assets` and then
invokes the mutable downloader; it does not consult `config.yaml`'s `assets`
entry (`external/hf-libero/libero/libero/__init__.py:68-97`). The proposed
source-free binding is therefore:

1. Materialize the exact asset revision into an empty, content-addressed
   directory outside the repository.
2. Verify the complete path set, sizes, and every LFS payload SHA256 against a
   recorded manifest; reject Git-LFS pointer text.
3. In the isolated installed environment only, create
   `site-packages/libero/libero/assets` as a symlink to that verified directory.
   Do not add anything to the external source checkout.
4. Verify the resolved symlink target and make `get_assets_path()` return it
   before constructing an environment.
5. Set Hub offline variables. Any missing path/hash aborts before `make_env`;
   the upstream downloader is never allowed to run.

### Exact-revision retrieval and load sequence

The proposed materialization commands are:

```bash
hf download HuggingFaceVLA/smolvla_libero \
  config.json model.safetensors policy_preprocessor.json \
  policy_preprocessor_step_5_normalizer_processor.safetensors \
  policy_postprocessor.json \
  policy_postprocessor_step_1_unnormalizer_processor.safetensors \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --local-dir /ARTIFACTS/smolvla-libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102

hf download HuggingFaceTB/SmolVLM2-500M-Instruct \
  config.json generation_config.json preprocessor_config.json \
  processor_config.json tokenizer_config.json tokenizer.json \
  special_tokens_map.json added_tokens.json chat_template.json \
  merges.txt vocab.json model.safetensors \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --local-dir /ARTIFACTS/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467

hf download lerobot/libero-assets --repo-type dataset \
  --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
  --local-dir /ARTIFACTS/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
```

After hash/size verification, loading must use only absolute local paths:

```python
config = SmolVLAConfig.from_pretrained(checkpoint_dir, local_files_only=True)
config.device = runtime_device
config.vlm_model_name = str(base_vlm_dir)  # locator only; in memory

policy = SmolVLAPolicy.from_pretrained(
    checkpoint_dir,
    config=config,
    local_files_only=True,
)

preprocessor, postprocessor = make_pre_post_processors(
    config,
    pretrained_path=str(checkpoint_dir),
    pretrained_revision="6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
    preprocessor_overrides={
        "tokenizer_processor": {"tokenizer_name": str(base_vlm_dir)},
        "device_processor": {"device": runtime_device},
    },
    postprocessor_overrides={
        "device_processor": {"device": "cpu"},
    },
)
```

The locator overrides do not alter the serialized tokenizer contents,
processor order, normalization map, or statistics. They close the pinned
v0.6.1 gap where base-model and tokenizer loaders do not forward a Hub
revision. Use an empty `HF_HOME` together with:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
```

## Runtime smoke results

| Gate | Result | Evidence |
| --- | --- | --- |
| Python isolation | **PASS** | Python 3.12.6 at `/public/home/xuyinghao/tmp/shiftvla-libero`; host Python/NumPy not used |
| Dependency consistency | **PASS (CPU resolution)** | `pip check`: `No broken requirements found.`; 139 frozen distributions |
| Source import identity | **PASS** | LeRobot, hf-libero and robosuite import from the isolated `site-packages`; their `direct_url.json` points to the exact local pinned source trees |
| HCU discovery | **FAIL** | no `/dev/dri`, `/dev/kfd`, `/dev/mkfd`; `hy-smi` reports no device |
| MuJoCo EGL render | **PASS, software only** | MuJoCo 3.7.0 produced finite `(64,64,3)` `uint8`; GL renderer `llvmpipe (LLVM 12.0.0, 256 bits)` |
| HCU EGL | **FAIL** | EGL context is Mesa software, not an HCU device |
| Official LIBERO config | **PASS** | `make_env_config("libero", task="libero_spatial", task_ids=[0])` gives 360×360, two cameras, `pixels_agent_pos`, `rgb_array` |
| LIBERO import/task lookup | **PASS** | suite task 0, BDDL, and initial-state file all resolved from the pinned tree |
| LIBERO reset/RGB render | **BLOCKED / NOT RUN** | exact asset directory absent; fail-closed gate exited 23 before environment construction |
| Selected config load | **PASS** | local-only, offline config loaded as `SmolVLAConfig`: state 8, two cameras, action 7, chunk 50, one queued action |
| Base processor/tokenizer metadata | **PASS** | local-only, offline `SmolVLMProcessor` and `GPT2Tokenizer` instantiated; sample instruction tokenized to `(1,6)` |
| Serialized policy processors | **BLOCKED AS EXPECTED** | LFS pointer caused `SafetensorError: ... header too large` under offline mode; no Hub fallback. This parser failure is not the production hash gate. |
| Full selected checkpoint load | **BLOCKED / NOT RUN** | verified payloads unavailable |
| One finite action forward | **BLOCKED / NOT RUN** | model load prerequisite unavailable; no random/substitute forward used |

If materialization and HCU gates later pass, the expected official full-chunk
output is `(1,50,7)` and `select_action` returns `(1,7)`. Internally the flow
state is `(1,50,32)`. These shapes are source/config verified, not runtime
forward evidence in this phase.

### 360×360 environment versus 256×256 policy metadata

This is not an automatic incompatibility and no environment resolution was
changed.

- The serialized policy preprocessor contains no resize operation.
- Generic LeRobot preprocessing converts HWC uint8 to NCHW float without
  resizing.
- `LiberoProcessorStep` rotates the images and forms 8-D proprioception, again
  without resizing.
- `SmolVLAPolicy.prepare_images` resizes/pads each present input to the
  configured 512×512 internal canvas
  (`modeling_smolvla.py:334-360`).

**Source-verified, runtime-unverified:** the official path is designed to
accept the registered 360×360 environment image and perform the model's own
512×512 resize. No real LIBERO observation reached that path in this preflight
because reset and model loading were blocked. The 256×256 feature shape is
metadata; it does not justify changing the environment to 256×256. Historical
numerical equivalence between 360 and the checkpoint's training capture remains
unknown and must be recorded in the eventual manifest.

## Material commands and probe outputs

### Environment creation and successful resolution

```bash
/usr/local/python3.12/bin/python3.12 -m venv \
  /public/home/xuyinghao/tmp/shiftvla-libero

PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt \
PIP_CACHE_DIR=/public/home/xuyinghao/tmp/shiftvla-pip-cache \
/public/home/xuyinghao/tmp/shiftvla-libero/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==2.11.0+cpu torchvision==0.26.0+cpu torchcodec==0.11.1+cpu \
  numpy==2.2.6 mujoco==3.7.0 transformers==5.5.4 \
  huggingface-hub==1.26.0 gymnasium==1.3.0 \
  opencv-python==4.14.0.94 opencv-python-headless==4.13.0.92 \
  scipy==1.18.0 numba==0.66.0 llvmlite==0.48.0 packaging==25.0 \
  setuptools==81.0.0 pandas==2.3.3 pyarrow==25.0.0 \
  datasets==4.8.5 fsspec==2026.2.0 protobuf==6.32.0 \
  wandb==0.27.2 typer==0.27.0 rich==15.0.0 requests==2.34.2 \
  /public/home/xuyinghao/workspace/vla/external/robosuite \
  /public/home/xuyinghao/workspace/vla/external/hf-libero \
  '/public/home/xuyinghao/workspace/vla/external/lerobot[smolvla,libero,evaluation]'

/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pip check
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pip freeze --all \
  > /public/home/xuyinghao/tmp/shiftvla-libero.freeze.txt

rg -v '^(#|$)' runtime/locks/shiftvla-libero-cpu-preflight.txt \
  > /public/home/xuyinghao/tmp/shiftvla-lock-packages.txt
diff -u /public/home/xuyinghao/tmp/shiftvla-libero.freeze.txt \
  /public/home/xuyinghao/tmp/shiftvla-lock-packages.txt
```

The first TLS attempt failed before installing anything because the standalone
Python 3.12 build's compiled default CA path did not exist. The retry used the
system CA file
`/etc/pki/tls/certs/ca-bundle.crt`
(SHA256 `887e33b65bef4e64c48bd3e9fe09c0e9066fe6c66429922dcdd61f3034dd1aba`),
not `--trusted-host`. An earlier torch 2.7.1 CPU solve was interrupted before
installation when unconstrained OpenCV drifted beyond the pinned LeRobot
resolution; the successful command above explicitly matched the v0.6.1 lock
generation except for deliberate CPU wheels and MuJoCo 3.7.0.

### Exact artifact download attempt

```bash
mkdir -p \
  /public/home/xuyinghao/tmp/shiftvla-artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 && \
SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
HF_HUB_DISABLE_TELEMETRY=1 \
/public/home/xuyinghao/tmp/shiftvla-libero/bin/hf download \
  lerobot/libero-assets --repo-type dataset \
  --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
  --local-dir /public/home/xuyinghao/tmp/shiftvla-artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
  --max-workers 8 --format quiet
```

The platform rejected this command before it ran because the escalation usage
quota was exhausted. Model/base downloads were consequently not attempted.

### EGL probe

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
XDG_CACHE_HOME=/public/home/xuyinghao/tmp/shiftvla-mesa-cache \
MESA_SHADER_CACHE_DIR=/public/home/xuyinghao/tmp/shiftvla-mesa-cache \
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python \
  /tmp/shiftvla_mujoco_egl_probe.py
```

Probe SHA256:
`9582a71bb0ba4c6b5f476d7282e7bee93c6f9572467fbaba55dfe28427bf0a26`.

```json
{"dtype":"uint8","gl_renderer":"llvmpipe (LLVM 12.0.0, 256 bits)","gl_vendor":"Mesa/X.org","gl_version":"3.1 Mesa 21.1.5","mujoco":"3.7.0","pixel_sum":47318,"shape":[64,64,3]}
```

### Offline metadata and fail-closed payload probes

```bash
HF_HOME=/public/home/xuyinghao/tmp/shiftvla-empty-hf-home \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python \
  /tmp/shiftvla_metadata_load_probe.py

HF_HOME=/public/home/xuyinghao/tmp/shiftvla-empty-hf-home \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python \
  /tmp/shiftvla_processor_payload_gate_probe.py
```

Probe SHA256 values are respectively
`5cf2d70477c25171fd3f355f10e1150e3efc41bce8010b92b31081add53d2370`
and
`8286e5dbf52616320f2225da5191a4e5f56fc233faa3b3e846c620578f21655e`.
The first returned the expected 8-D/two-camera/7-D/50/1 contract and loaded
the local processor/tokenizer. The second returned:

```json
{"blocked_as_expected":true,"exception":"SafetensorError","message":"Error while deserializing header: header too large"}
```

### Vanilla LIBERO artifact gate

The pre-created config was
`/public/home/xuyinghao/tmp/shiftvla-libero-config/config.yaml`, SHA256
`98d57e0d3d7b70bab5c66a110ecc333c737895e9dcd457c06ec94b4150e0ecc8`:

```yaml
benchmark_root: /public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero
bddl_files: /public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/bddl_files
init_states: /public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/init_files
datasets: /public/home/xuyinghao/tmp/shiftvla-libero-datasets
assets: /public/home/xuyinghao/tmp/shiftvla-artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
```

```bash
LIBERO_CONFIG_PATH=/public/home/xuyinghao/tmp/shiftvla-libero-config \
HF_HOME=/public/home/xuyinghao/tmp/shiftvla-empty-hf-home \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python \
  /tmp/shiftvla_libero_asset_gate_probe.py
```

Probe SHA256:
`f4651586e5e8f85f431ad715ed113c5b9a66767ab700ebd5b36944f141decaaf`.
It exited 23 before environment construction and reported that the task, BDDL,
and init state existed while the exact asset directory did not.

### Source and worktree verification

```bash
git -C external/lerobot rev-parse HEAD
git -C external/hf-libero rev-parse HEAD
git -C external/robosuite rev-parse HEAD
git -C external/mujoco rev-parse HEAD
git -C external/lerobot status --porcelain
git -C external/hf-libero status --porcelain
git -C external/robosuite status --porcelain
git -C external/mujoco status --porcelain
```

The four HEADs matched `external/pins.yaml`; all status outputs were empty.

## Unresolved risks and approval gates

1. **Checkpoint/base payloads unavailable.** Full model load and one finite
   forward remain mandatory before M0 approval.
2. **Vanilla asset payload unavailable.** Approve the proposed
   `lerobot/libero-assets@0b3ea86…` pin and materialize/verify it before the
   first reset/render.
3. **HCU runtime unavailable.** A Python 3.12-compatible DTK PyTorch stack and
   visible HCU device nodes are required. CPU and llvmpipe success do not meet
   this gate.
4. **Historical provenance incomplete.** The retained checkpoint's historical
   LeRobot, LIBERO, training dataset, and base-model revisions are unknown.
5. **Base loader revision gap.** Local absolute paths plus offline mode are the
   proposed fail-closed seam; the pinned loader itself does not forward a base
   VLM/tokenizer revision.
6. **hf-libero asset resolver gap.** Its config `assets` key is ignored. The
   proposed isolated-environment symlink must be reviewed and verified before
   use; no source patch is proposed.
7. **OpenCV namespace collision.** Pinned LeRobot requires headless OpenCV while
   hf-libero requires full OpenCV. The resolved environment contains both and
   currently imports 4.14.0. This matches the upstream dependency graph but
   remains a packaging risk.
8. **Lock strength.** The tracked file records the exact successful package
   resolution, not hashes for every Python wheel or a working HCU wheel set.
   A hash-complete accepted HCU lock is still required before M0.
9. **EGL device discrimination.** robosuite can accept software EGL. The future
   M0 preflight must explicitly require the intended HCU renderer and abort on
   llvmpipe; it must not switch to OSMesa.
10. **Image provenance.** The official 360→512 path is source verified, but the
    checkpoint's historical training capture resolution is not recorded.

M0 baseline rollouts must not begin until the artifact, reset/render, HCU, full
model-load, and finite-forward gates above pass and this preflight receives
human review.

## M0-PREFLIGHT-B addendum (2026-08-25)

### Status

**PASS (main-reviewed).** The exact requested payloads, vanilla LIBERO
reset/RGB render, strict local-only SmolVLA load, serialized processors, and
one real observation-to-finite-action forward all passed in the isolated
Python 3.12 CPU environment. The executor's earlier `DONE_WITH_CONCERNS`
status and HCU/performance concerns are retained as provenance; main review
accepted the B gates. The only available EGL context is Mesa `llvmpipe` and
no HCU device is visible, so HCU acceptance is not claimed and remains an
unresolved performance/runtime risk. There was no rollout, replay,
perturbation, activation intervention, matched sampling experiment, or
training in this addendum.

The Preflight-A evidence above is retained. B added exactly one dependency
identity to `external/pins.yaml`: `lerobot/libero-assets` at the approved
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` revision. No existing pin identity
was edited.

### Baseline and scope verification

At B start, the repository HEAD was
`1c9cf2b4a2e441183c750818e3d21669f7f39358`. The pre-existing `AGENTS.md`
change was preserved byte-for-byte (SHA256
`edb351e926097b9b6a39b3d6900c48797e7d666bd132e2ad48b6ab711920314b`). The
four source worktrees remained detached, clean, and at the pinned revisions:

| Source | Revision | Nested status |
| --- | --- | --- |
| LeRobot | `7e241bd630a3719a56157a497ce5d08f244784f1` | clean |
| hf-libero | `8561c60eea2fb93096146f240194649df73d8b1e` | clean |
| robosuite | `fbee5844ff5632f5b5698e204ec5357ca50be0df` | clean |
| MuJoCo source | `72cb2b210da666617924de709406d6aadbe60c71` | clean |

The selected checkpoint and base-model source metadata worktrees also remained
clean at the frozen SHAs. The project-local artifact directories are ignored
external materialization paths; their complete inventory and per-file hashes
are recorded in
`runtime/manifests/m0_preflight_b_artifacts.json`.

### Exact pin and materialization

The new pin records the official URL
`https://huggingface.co/datasets/lerobot/libero-assets`, resolved ref
`refs/heads/main` →
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`, retrieval date `2026-08-25`,
purpose, compatibility notes, absolute materialization path, and expected and
observed materialized/clean status.

The first normal-sandbox command failed before transfer with
`httpx.ConnectError: [Errno 1] Operation not permitted`. One platform-
authorized exact command then downloaded the three required revisions. No
mutable branch, main/latest fallback, mirror, cache substitute, or repeated
materialization attempt was used. The commands were:

```bash
hf download HuggingFaceVLA/smolvla_libero \
  config.json model.safetensors policy_preprocessor.json \
  policy_preprocessor_step_5_normalizer_processor.safetensors \
  policy_postprocessor.json \
  policy_postprocessor_step_1_unnormalizer_processor.safetensors \
  --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
  --local-dir /public/home/xuyinghao/workspace/vla/external/artifacts/smolvla_libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102

hf download HuggingFaceTB/SmolVLM2-500M-Instruct \
  config.json generation_config.json preprocessor_config.json \
  processor_config.json tokenizer_config.json tokenizer.json \
  special_tokens_map.json added_tokens.json chat_template.json \
  merges.txt vocab.json model.safetensors \
  --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --local-dir /public/home/xuyinghao/workspace/vla/external/artifacts/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467

hf download lerobot/libero-assets --repo-type dataset \
  --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
  --local-dir /public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
```

Materialization and fail-closed verification results:

| Repository | Files | Total bytes | LFS files/bytes | Payload verification |
| --- | ---: | ---: | ---: | --- |
| `lerobot/libero-assets` | 586 | 422,320,936 | 177 / 325,067,554 | all sizes and LFS SHA256 values match exact tree metadata |
| `HuggingFaceVLA/smolvla_libero` | 6 | 1,218,052,342 | 3 / 1,218,047,864 | all required files and LFS SHA256 values match |
| `HuggingFaceTB/SmolVLM2-500M-Instruct` | 12 | 2,034,845,165 | 1 / 2,029,990,624 | all required files and model LFS SHA256 match |

The runtime-critical object hashes include checkpoint
`model.safetensors` SHA256
`71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`
(1,218,047,032 bytes), both serialized normalizer/unnormalizer state files
SHA256 `7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e`
(416 bytes each), and base-model `model.safetensors` SHA256
`b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3`
(2,029,990,624 bytes). The asset tree's 177 LFS payloads were all hashed;
zero pointer files and zero incomplete files remained. The manifest contains
the SHA256, byte count, exact tree metadata path/revision, and required
filename inventory for every materialized runtime file.

### Exact B commands executed

The following records use the exact executable paths, revisions, repo types,
absolute materialization paths, and offline environment used for B. The first
network attempt was the single normal-sandbox command shown below; it failed
with the recorded sandbox network error. The one platform-authorized command
then ran the three exact-revision downloads in order:

```bash
mkdir -p /public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 && /public/home/xuyinghao/tmp/shiftvla-libero/bin/hf download lerobot/libero-assets --repo-type dataset --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --local-dir /public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --format quiet
```

Result: `httpx.ConnectError: [Errno 1] Operation not permitted` before
transfer.

```bash
mkdir -p /public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 /public/home/xuyinghao/workspace/vla/external/artifacts/smolvla_libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102 /public/home/xuyinghao/workspace/vla/external/artifacts/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467 && HF_HOME=/public/home/xuyinghao/tmp/shiftvla-m0b-hf-home SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt HF_HUB_DISABLE_TELEMETRY=1 /public/home/xuyinghao/tmp/shiftvla-libero/bin/hf download HuggingFaceVLA/smolvla_libero config.json model.safetensors policy_preprocessor.json policy_preprocessor_step_5_normalizer_processor.safetensors policy_postprocessor.json policy_postprocessor_step_1_unnormalizer_processor.safetensors --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 --local-dir /public/home/xuyinghao/workspace/vla/external/artifacts/smolvla_libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102 --format quiet && HF_HOME=/public/home/xuyinghao/tmp/shiftvla-m0b-hf-home SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt HF_HUB_DISABLE_TELEMETRY=1 /public/home/xuyinghao/tmp/shiftvla-libero/bin/hf download HuggingFaceTB/SmolVLM2-500M-Instruct config.json generation_config.json preprocessor_config.json processor_config.json tokenizer_config.json tokenizer.json special_tokens_map.json added_tokens.json chat_template.json merges.txt vocab.json model.safetensors --revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467 --local-dir /public/home/xuyinghao/workspace/vla/external/artifacts/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467 --format quiet && HF_HOME=/public/home/xuyinghao/tmp/shiftvla-m0b-hf-home SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt HF_HUB_DISABLE_TELEMETRY=1 /public/home/xuyinghao/tmp/shiftvla-libero/bin/hf download lerobot/libero-assets --repo-type dataset --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --local-dir /public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 --format quiet
```

The exact artifact verification command was:

```bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python /tmp/shiftvla_build_artifact_manifest.py
```

It rebuilt
`runtime/manifests/m0_preflight_b_artifacts.json` and failed closed on missing
tree metadata, inventory drift, size/SHA256 mismatch, incomplete files, or
Git-LFS pointer text. The environment-local asset binding command was:

```bash
VENV=/public/home/xuyinghao/tmp/shiftvla-libero; SITE=$($VENV/bin/python - <<'PY'
import sysconfig
print(sysconfig.get_paths()['purelib'])
PY
); TARGET=/public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6; LINK=$SITE/libero/libero/assets; test ! -e "$LINK" && test ! -L "$LINK"; ln -s "$TARGET" "$LINK"; readlink -e "$LINK"; test "$(readlink -e "$LINK")" = "$TARGET"
```

The final fresh strict offline reset/render/model/one-forward probe command
was:

```bash
test ! -e /public/home/xuyinghao/tmp/shiftvla-empty-hf-home-g && mkdir -p /public/home/xuyinghao/tmp/shiftvla-empty-hf-home-g /public/home/xuyinghao/tmp/shiftvla-mplconfig-g /public/home/xuyinghao/tmp/shiftvla-xdg-g && LIBERO_CONFIG_PATH=/public/home/xuyinghao/tmp/shiftvla-libero-config HF_HOME=/public/home/xuyinghao/tmp/shiftvla-empty-hf-home-g HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 MPLCONFIGDIR=/public/home/xuyinghao/tmp/shiftvla-mplconfig-g XDG_CACHE_HOME=/public/home/xuyinghao/tmp/shiftvla-xdg-g /public/home/xuyinghao/tmp/shiftvla-libero/bin/python /tmp/shiftvla_m0_preflight_b_probe.py > /tmp/shiftvla_m0_preflight_b_probe_final_g.stdout 2> /tmp/shiftvla_m0_preflight_b_probe_final_g.stderr
```

The probe uses absolute local checkpoint/base paths, `strict=True`, the
ordinary full-chunk API, `policy.eval()`, and `torch.inference_mode()`; it does
not call `env.step`. The captured result JSON SHA256 was
`cd5d2ff6e4479969e94af2f359f13d2c339d7fd05045562a5055ab37956b00a1`. The
exact environment consistency commands were:

```bash
PIP_CACHE_DIR=/public/home/xuyinghao/tmp/shiftvla-pip-cache /public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pip check
PIP_CACHE_DIR=/public/home/xuyinghao/tmp/shiftvla-pip-cache /public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pip freeze --all > /tmp/shiftvla-runtime-freeze-now.txt
```

### Vanilla LIBERO reset and render

The exact asset directory was bound only in the isolated environment by the
reversible symlink:

```text
/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/libero/libero/assets
  -> /public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
```

`LIBERO_CONFIG_PATH` pointed to the pinned hf-libero BDDL/init tree. With
`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and the symlink already present,
`get_assets_path()` resolved to the verified directory and did not invoke the
mutable downloader. The official LeRobot vanilla LIBERO config used suite
`libero_spatial`, task ID `0`, `pixels_agent_pos`, two cameras, and the
registered 360×360 observation resolution. The BDDL and init-state paths were:

```text
/public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl
/public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init
```

Observed reset/render evidence (the probe's caller did not invoke public
`env.step`):

| Field | Result |
| --- | --- |
| raw observation images | two finite `uint8` arrays, each `(1, 360, 360, 3)` |
| raw robot state | finite float64 nested state: eef pos `(1,3)`, quat `(1,4)`, gripper qpos `(1,2)`, joints pos `(1,7)` |
| RGB render | finite `uint8` `(360,360,3)` |
| `MUJOCO_GL` / device ID | `egl` / `0` |
| EGL vendor / renderer / version | Mesa/X.org / `llvmpipe (LLVM 12.0.0, 256 bits)` / `3.1 Mesa 21.1.5` |
| render timing sample | `0.00014554`, `0.00011261`, `0.00011246` seconds |

The standard pinned `LiberoEnv.reset()` settling behavior is part of reset;
there was no caller-level step, rollout, or action application. The llvmpipe
renderer is a software performance warning and fails the separate HCU runtime
acceptance boundary.

### Offline local-only model and one real forward

The final fresh probe used the isolated environment at
`/public/home/xuyinghao/tmp/shiftvla-libero` with an empty
`HF_HOME=/public/home/xuyinghao/tmp/shiftvla-empty-hf-home-g` and:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
```

`config.vlm_model_name` and the serialized tokenizer processor locator were
set to the absolute local base-model path in memory. The serialized processor
order, schema, normalization mapping, and state statistics were loaded from
the exact local checkpoint files without schema repair or source changes.
`SmolVLAPolicy.from_pretrained` loaded with `strict=True`; `policy.eval()` was
true and the forward ran under `torch.inference_mode()`.

The one-observation path was exactly:

```text
vanilla LIBERO reset observation
  -> official preprocess_observation
  -> LiberoProcessorStep
  -> serialized policy preprocessor
  -> frozen SmolVLAPolicy (local checkpoint + local SmolVLM base)
  -> serialized policy postprocessor
  -> first action from the returned action chunk
```

The observed feature and action evidence was:

| Stage | Evidence |
| --- | --- |
| official environment processor | images `(1,3,360,360)` float32; state `(1,8)` float32 |
| serialized policy preprocessor | images `(1,3,360,360)` float32; state `(1,8)` float32; tokens `(1,20)` int64; attention mask `(1,20)` bool |
| model's official image preparation | two images `(1,3,512,512)` float32; padded state `(1,32)` float32 |
| frozen model output chunk | finite float32 `(1,50,7)` |
| postprocessed first action | finite float32 `(1,7)`, min `-0.9770377278`, max `0.2601070106`, all values within `[-1,1]` |

The emitted action was
`[0.1609623283, 0.0820431486, 0.2601070106, -0.0019907039,
0.0480544530, 0.0088271555, -0.9770377278]`. The ordinary full-chunk API
call used global PyTorch seed `0` for the action-generation noise; this was
not a matched-noise or paired-comparison test (`noise_pairing_test: false`).
No optimizer, gradient, training, LoRA, hook, or activation patching path was
entered.

The loaded policy contained `604934176` total parameters. The effective
trainable count for this inference-only run was `0`: no optimizer, backward,
or gradient computation was performed. The observed `requires_grad` flags
remained unchanged, with `97451872` parameter elements across `299` tensors
still flagged `requires_grad`; this is an observation of serialized/runtime
flags, not a claim that training occurred. All `788` parameter tensors were
on CPU. Parameter tensors were `torch.bfloat16` (`746` tensors,
`600902304` elements) or `torch.float32` (`42` tensors, `4031872` elements).
The runtime versions were torch `2.11.0+cpu`, transformers `5.5.4`, NumPy
`2.2.6`, torch CUDA `None`, torch HIP `None`, and hipcc `25.08.0-0` from
`/opt/dtk-25.04.2/bin/hipcc`.

The 360×360 → model-internal 512×512 path was therefore exercised through
the official model preparation. The serialized feature metadata still names
256×256 policy feature shapes; the environment resolution was not changed and
no manual resize or schema adaptation was introduced.

### Runtime lock and remaining boundary

The exact environment used by the successful reset/render and forward is
recorded in
`runtime/locks/shiftvla-libero-runtime.txt`: Python 3.12.6, torch
2.11.0+cpu, transformers 5.5.4, MuJoCo 3.7.0, the complete `pip freeze
--all`, `pip check` output, source SHAs, offline variables, EGL metadata, and
the CPU/HCU boundary. `pip check` returned `No broken requirements found.`

M0 remains **NOT_STARTED** and is not automatically authorized by this phase;
wait for the next explicit instruction. CPU reset/render and a CPU forward
establish software/runtime correctness only; they do not establish device
performance, HCU bfloat16 execution, or pilot acceptance. CPU/llvmpipe is
recorded and no HCU acceptance is claimed. No baseline rollout may start from
this addendum.
