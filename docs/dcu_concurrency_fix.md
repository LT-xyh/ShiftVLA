# DCU concurrency runtime validation

Status: PASS for the frozen two-worker DCU runtime gate. This document records
the evidence-only closure of `DCU-CONCURRENCY-FIX`; it does not claim M0
baseline acceptance. The immutable input-lock bytes are archived at
`runtime/locks/archive/shiftvla-libero-dcu-runtime.c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a.txt`.

## Root cause and exact fix

The concurrency parent passed `_worker_environment(config, device)` directly
to each CPU LIBERO child. That helper is intentionally responsible for the
nested DCU model worker and sets `HIP_VISIBLE_DEVICES` to the assigned
physical device and `CUDA_VISIBLE_DEVICES=0`. The variables leaked into the
CPU child before robosuite initialized EGL. Robosuite then compared the
independent `MUJOCO_EGL_DEVICE_ID=8` against the CUDA visibility string and
raised its assertion. CUDA/HIP compute-device IDs and MuJoCo EGL-device IDs
are separate namespaces.

The fix adds `build_cpu_child_environment`: it copies the validated offline,
cache, LIBERO, and renderer environment, removes `HIP_VISIBLE_DEVICES` and
`CUDA_VISIBLE_DEVICES` from the CPU child, and preserves explicit EGL ordinal
8. `_start_worker` still creates the nested model-worker environment with
physical K100 0 or 1 mapped to `HIP_VISIBLE_DEVICES=0` or `1`,
`CUDA_VISIBLE_DEVICES=0`, and logical `cuda:0`. The child validates the actual
EGL enumeration (`eglQueryDevicesEXT` count 9), ordinal 8, and exact
llvmpipe identity. No GPU EGL optimization was attempted.

The child boundary is fixed at one reset, two official policy decisions, and
two real `env.step` calls. Parent validation is fail-closed over model load,
offline/no-Hub-fallback evidence, device mappings, EGL/GL evidence, finite
float32 `[1,7]` actions, latency, memory, and terminal child manifests.
Children use their own process groups; failure, timeout, or termination
terminalizes every created child manifest and cleans up only those groups.

## Validation input and exact command

The implementation was executed at project SHA
`38517cbcac9f08154fe93bfe9e25a650c464ee45`. At smoke time, the validated
input-lock bytes were read from
`runtime/locks/shiftvla-libero-dcu-runtime.txt`. Those exact bytes were later
materialized at
`runtime/locks/archive/shiftvla-libero-dcu-runtime.c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a.txt`,
with SHA
`c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a`.
The evidence-closure lock now has SHA
`cc507d48c64e72a9d2f6552bc5fa638217f9e6a2addef0067ded3d0adce30ed2`; the
original input-lock SHA is retained explicitly in that lock.
The input freeze remained 139 lines and SHA
`aa980a6cae3c6e7aea4c80f0f8086184d05903aae1140374f87dd5b6b4b8601a`.

The only real concurrency smoke command was:

```text
env HF_HOME=/tmp/shiftvla-concurrency-hf.8e7mRk MPLCONFIGDIR=/tmp/shiftvla-concurrency-mpl.MyBdJQ XDG_CACHE_HOME=/tmp/shiftvla-concurrency-xdg.Y6l1sK PIP_CACHE_DIR=/tmp/shiftvla-concurrency-pip.Jdcukr HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 LIBERO_CONFIG_PATH=/public/home/xuyinghao/tmp/shiftvla-libero-config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=8 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /public/home/xuyinghao/tmp/shiftvla-libero/bin/python scripts/dcu_preflight.py concurrency --config configs/m0/dcu_preflight.yaml --expected-project-sha 38517cbcac9f08154fe93bfe9e25a650c464ee45
```

Run directory:
`/public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260828T024505Z`

## Acceptance evidence

| Gate | Evidence |
| --- | --- |
| A: distinct physical devices | Worker A used physical K100 0; worker B used physical K100 1. |
| B: model load | Both `model_load_success=true`; model peak memory was 2,483,390,464 bytes per worker. |
| C: reset/render | Both reset count 1 and render count 1; EGL count 9, selected ordinal 8, `test_device` return code 0. |
| D: real steps | Both made 2 policy decisions and 2 real environment steps. |
| E: actions | Each worker emitted 2 finite float32 actions of shape `[1,7]`. |
| F: offline | `offline_execution=true`, `hub_fallback=false`. |
| G: manifests | Parent and both children ended `PASS`; child manifests are explicit terminal records with return code 0. |
| H: semantics | No checkpoint, processor, precision, renderer identity, dependency, or scientific-runtime change was made in this closure. Attention remains the existing eager runtime setting; no attention change was made. Full 280-step episode and M0 baseline were not run. |

## Device and EGL mapping

| Worker | Physical K100 | CPU child HIP/CUDA | Nested HIP/CUDA | Torch logical device/name |
| --- | ---: | --- | --- | --- |
| A | 0 | UNSET / UNSET | 0 / 0 | `cuda:0` / `K100_AI` |
| B | 1 | UNSET / UNSET | 1 / 0 | `cuda:0` / `K100_AI` |

Both children recorded the same independently selected EGL mapping:

```text
eglQueryDevicesEXT count: 9
MUJOCO_EGL_DEVICE_ID: 8
egl test_device return code: 0
GL_VENDOR: Mesa/X.org
GL_RENDERER: llvmpipe (LLVM 12.0.0, 256 bits)
GL_VERSION: 3.1 Mesa 21.1.5
```

## Per-worker performance

| Worker | Model load (s) | Inference (s) | Env step (s) | Worker wall (s) | Model peak (bytes) | Action peak (bytes) |
| --- | ---: | --- | --- | --- | ---: | ---: |
| A | 11.4003563467 | `[2.3507065307, 0.6460360456]` | `[0.2640677262, 0.2833882552]` | `[2.2660737969, 0.6108500846]` | 2483390464 | 1423232512 |
| B | 12.5096133426 | `[2.2906814069, 0.6443344429]` | `[0.2823194042, 0.2700402047]` | `[2.2248826101, 0.6036770921]` | 2483390464 | 1423232512 |

Concurrent wall time was `59.5112689249` seconds. Throughput was:

```text
total_env_steps              4
total_policy_decisions       4
env_steps_per_second         0.0672141608
policy_decisions_per_second  0.0672141608
workers_per_second           0.0336070804
```

## Artifact hashes

```text
parent run_manifest.json: 6a59f62c0714a397486d126029d13e3aeb84ecaa450153450baedd7de8d9cf9c
concurrency.json:          a997aefa945e7fca83c8d118d02fd35ca89d71d671353445b110e16fda22c8ae
child A run_manifest.json: 6bf4bce51dd761261f29c4f9528935b1a7f0e694ebdfbc03f5f628a376c043e9
child B run_manifest.json: b92e031b76482f3b231e6d56fe5ce7c28610536e03f88e5624d7e2f8c5e66b87
child A result:            80b3e0d9962fbab5fc2a25002095aef66079436b54e1e5ba17799363425ecd65
child B result:            add1b7cc3a55600e7dccdc0f7f053304c862810955f35ab520a2aa3e7f1a5632
```

The smoke emitted the known NumPy ABI warning: the torch extension was built
against NumPy 1.x while the environment has NumPy 2.2.6 (`_ARRAY_API not
found`). This was recorded as a warning only; no dependency or runtime
change was made.

This closure records exactly two workers and two steps per worker. It does
not close or start M0 baseline, and it does not establish full-episode
scientific acceptance.
