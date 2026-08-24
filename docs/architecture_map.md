# ShiftVLA Architecture Map

## Phase boundary and evidence labels

**VERIFIED FROM PINNED SOURCE**

This document is a read-only static audit of immutable source and metadata
checkouts. The ShiftVLA specification anchor is
53ab983e6fa32f26d5d223927810be022ff5bc6f. The dependency manifest is
external/pins.yaml, retrieved on 2026-08-24.

**NOT YET VERIFIED**

No Python environment was installed, no LFS payload was materialized, no model
was loaded, no transformer or hook probe was run, and no simulator was reset or
rendered. Statements about runtime tensor values, numerical equivalence, hook
firing, EGL, checkpoint compatibility, or exact cross-environment replay remain
unverified even when their intended code path is statically clear.

**PROPOSED DESIGN**

The replay, activation-intervention, and M0 sections describe future seams only.
No M0, replay, hook, activation-patching, or experiment code exists in this
bootstrap milestone.

## Immutable inventory

**VERIFIED FROM PINNED SOURCE**

All source locations below are ordinary detached Git checkouts, not submodules.
Every checkout was clean at audit time.

| ID | Absolute local root | Official origin | Revision | Resolved from |
| --- | --- | --- | --- | --- |
| LeRobot | /public/home/xuyinghao/workspace/vla/external/lerobot | https://github.com/huggingface/lerobot.git | 7e241bd630a3719a56157a497ce5d08f244784f1 | tag v0.6.1 |
| canonical LIBERO | /public/home/xuyinghao/workspace/vla/external/libero | https://github.com/Lifelong-Robot-Learning/LIBERO.git | 8f1084e3132a39270c3a13ebe37270a43ece2a01 | master |
| LeRobot runtime LIBERO | /public/home/xuyinghao/workspace/vla/external/hf-libero | https://github.com/huggingface/LIBERO.git | 8561c60eea2fb93096146f240194649df73d8b1e | main |
| LIBERO-Plus | /public/home/xuyinghao/workspace/vla/external/libero-plus | https://github.com/sylvestf/LIBERO-plus.git | 4976dc30028e805ff8094b55501d532c48fec182 | main |
| robosuite | /public/home/xuyinghao/workspace/vla/external/robosuite | https://github.com/ARISE-Initiative/robosuite.git | fbee5844ff5632f5b5698e204ec5357ca50be0df | tag v1.4.0 |
| MuJoCo | /public/home/xuyinghao/workspace/vla/external/mujoco | https://github.com/google-deepmind/mujoco.git | 72cb2b210da666617924de709406d6aadbe60c71 | tag 3.7.0 |
| SmolVLA checkpoint metadata | /public/home/xuyinghao/workspace/vla/external/models/smolvla-libero | https://huggingface.co/HuggingFaceVLA/smolvla_libero | 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 | main |
| base SmolVLM metadata | /public/home/xuyinghao/workspace/vla/external/models/smolvlm2-500m-instruct | https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct | 7b375e1b73b11138ff12fe22c8f2822d8fe03467 | main |
| Plus asset metadata | /public/home/xuyinghao/workspace/vla/external/datasets/libero-plus-assets | https://huggingface.co/datasets/Sylvest/LIBERO-plus | b548dd25ee0401c46217ba7e3614a598e8979e48 | historical commit from main |

**VERIFIED FROM PINNED SOURCE**

The selected checkpoint model.safetensors LFS object is
sha256:71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f
(1,218,047,032 bytes). Its preprocessor and postprocessor normalization state
files both point to
sha256:7008ba73c73887e5a7e631d478de9563ec5a0c51b7f0327f2530d7326c4b039e
(416 bytes). The Plus assets.zip pointer records
sha256:96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
(6,395,849,578 bytes). These are pointer-verified LFS content OIDs; the payloads
were intentionally not downloaded.

**NOT YET VERIFIED**

The recorded hf-libero 0.1.4 wheel digest is
207f76e2f28bff30f78132223d8592fe8f64b1f8fd90ce7024948ada0d7e2c27,
but the wheel is not materialized or installed in this phase. Plus asset
contents, BDDL completeness, initial-state payload completeness, and model
weights are likewise not locally payload-verified.

### Selected and rejected checkpoint contracts

**VERIFIED FROM PINNED SOURCE**

The selected HuggingFaceVLA/smolvla_libero revision declares:

- two image inputs, observation.images.image and
  observation.images.image2, each shaped (3, 256, 256);
- observation.state shaped (8);
- action shaped (7);
- chunk_size 50, n_action_steps 1;
- max_state_dim 32, max_action_dim 32;
- resize_imgs_with_padding (512, 512);
- tokenizer_max_length 48, num_steps 10, and use_cache true;
- HuggingFaceTB/SmolVLM2-500M-Instruct as the named VLM;
- all 32 VLM layers and all 32 expert layers, cross-attention mode,
  self-attention every second expert layer, and expert width multiplier 0.5.

Evidence: /public/home/xuyinghao/workspace/vla/external/models/smolvla-libero/config.json,
lines 1-87.

**NOT YET VERIFIED**

The candidate lerobot/smolvla_libero at
31d453f7edd78c839a8bbc39744a292686daf0de was rejected during bootstrap
because it declares three cameras and 6D state. It must not be substituted for
the selected two-camera/8D checkpoint, and no camera synthesis or state
truncation is permitted. This rejection is recorded from the approved
bootstrap evidence, but that candidate is deliberately not one of the local
pinned checkouts and is therefore not evidence from a pinned tree.

## Pinned-source index

Each row gives the exact absolute file path, its repository-relative path, the
relevant line range and symbol, and the revision from which the statement was
read.

| Status | Absolute path | Repository-relative path and lines | Class/function | Revision |
| --- | --- | --- | --- | --- |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/scripts/lerobot_eval.py | src/lerobot/scripts/lerobot_eval.py:167-307, 414-549, 739-820 | rollout, eval_policy, eval_main | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/configs/eval.py | src/lerobot/configs/eval.py:29-79 | EvalPipelineConfig.__post_init__ | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/envs/factory.py | src/lerobot/envs/factory.py:37-122 | make_env_pre_post_processors, make_env | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/envs/configs.py | src/lerobot/envs/configs.py:320-451, 729-750 | LiberoEnv, LiberoPlusEnv | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/policies/factory.py | src/lerobot/policies/factory.py:150-229, 240-386 | make_pre_post_processors, make_policy | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/configs/policies.py | src/lerobot/configs/policies.py:171-236 | PreTrainedConfig.from_pretrained | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/policies/pretrained.py | src/lerobot/policies/pretrained.py:168-227 | PreTrainedPolicy.from_pretrained | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/envs/utils.py | src/lerobot/envs/utils.py:68-120 | preprocess_observation | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/processor/env_processor.py | src/lerobot/processor/env_processor.py:26-112 | LiberoProcessorStep | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/processor/pipeline.py | src/lerobot/processor/pipeline.py:618-660 | PolicyProcessorPipeline.from_pretrained | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/processor/tokenizer_processor.py | src/lerobot/processor/tokenizer_processor.py:54-116 | TokenizerProcessorStep | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/models/smolvla-libero/config.json | config.json:1-87 | serialized SmolVLAConfig | 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/models/smolvla-libero/policy_preprocessor.json | policy_preprocessor.json:1-79 | serialized policy preprocessor | 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/models/smolvla-libero/policy_postprocessor.json | policy_postprocessor.json:1-32 | serialized policy postprocessor | 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/models/smolvlm2-500m-instruct/config.json | config.json:1-140 | serialized SmolVLMConfig | 7b375e1b73b11138ff12fe22c8f2822d8fe03467 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py | src/lerobot/policies/smolvla/modeling_smolvla.py:142-269, 334-418, 492-519, 546-810 | SmolVLAPolicy, VLAFlowMatching | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/policies/common/flow_matching.py | src/lerobot/policies/common/flow_matching.py:35-122 | sample_noise, euler_integrate | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py | src/lerobot/policies/smolvla/smolvlm_with_expert.py:67-215, 220-394, 396-509 | SmolVLMWithExpertModel | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/src/lerobot/envs/libero.py | src/lerobot/envs/libero.py:34-100, 107-190, 258-391, 403-520 | LiberoEnv and create_libero_envs | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/envs/env_wrapper.py | libero/libero/envs/env_wrapper.py:12-145 | ControlEnv | 8561c60eea2fb93096146f240194649df73d8b1e |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/hf-libero/libero/libero/envs/venv.py | libero/libero/envs/venv.py:211-290, 884-921, 924-969 | worker/vector command protocol | 8561c60eea2fb93096146f240194649df73d8b1e |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/robosuite/robosuite/utils/binding_utils.py | robosuite/utils/binding_utils.py:213-241, 1082-1092, 1140-1169 | MjSimState, MjSim | fbee5844ff5632f5b5698e204ec5357ca50be0df |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/mujoco/include/mujoco/mjdata.h | include/mujoco/mjdata.h:27-52 | mjtState, mjSTATE_INTEGRATION | 72cb2b210da666617924de709406d6aadbe60c71 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/mujoco/include/mujoco/mujoco.h | include/mujoco/mujoco.h:485-501 | mj_stateSize, mj_getState, mj_setState | 72cb2b210da666617924de709406d6aadbe60c71 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/mujoco/python/mujoco/functions.cc | python/mujoco/functions.cc:325-359 | mj_stateSize/getState/setState Python bindings | 72cb2b210da666617924de709406d6aadbe60c71 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/mujoco/doc/programming/simulation.rst | doc/programming/simulation.rst:376-397, 1251-1266 | integration/simulation state and sleeping caveat | 72cb2b210da666617924de709406d6aadbe60c71 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/libero/libero/envs/env_wrapper.py | libero/libero/envs/env_wrapper.py:176-270, 315-390 | ControlEnv filename parser and state wrapper | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/libero/libero/envs/bddl_base_domain.py | libero/libero/envs/bddl_base_domain.py:298-392 | BDDLBaseDomain._load_model | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/libero/libero/envs/problems/libero_tabletop_manipulation.py | libero/libero/envs/problems/libero_tabletop_manipulation.py:11-45, 305-351 | scale_distance_from_pivot, _setup_camera | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/libero/libero/benchmark/__init__.py | libero/libero/benchmark/__init__.py:80-111 | task map and task counts | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/libero/libero/benchmark/libero_suite_task_map.py | libero/libero/benchmark/libero_suite_task_map.py:1-10131 | 10120 task-name map | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/libero/libero/benchmark/task_classification.json | libero/libero/benchmark/task_classification.json:1-60189 | four-suite 10030-record classification | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/README.md | README.md:24-33 | stated 10030-task/FOV scope | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/robosuite/robosuite/models/arenas/arena.py | robosuite/models/arenas/arena.py:46-72 | Arena.set_camera | fbee5844ff5632f5b5698e204ec5357ca50be0df |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/robosuite/robosuite/models/base.py | robosuite/models/base.py:125-147 | MujocoXML.get_model | fbee5844ff5632f5b5698e204ec5357ca50be0df |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/pyproject.toml | pyproject.toml:26-78, 152-152, 267-272 | package and runtime constraints | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/uv.lock | uv.lock:3961-3962 | locked MuJoCo version | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/hf-libero/pyproject.toml | pyproject.toml:5-40 | hf-libero 0.1.4 runtime constraints | 8561c60eea2fb93096146f240194649df73d8b1e |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/setup.py | setup.py:6-16 | libero 0.1.0 package metadata | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/libero-plus/requirements.txt | requirements.txt:1-16 | legacy Plus constraints | 4976dc30028e805ff8094b55501d532c48fec182 |
| VERIFIED FROM PINNED SOURCE | /public/home/xuyinghao/workspace/vla/external/lerobot/docker/Dockerfile.benchmark.libero_plus | docker/Dockerfile.benchmark.libero_plus:34-79 | LeRobot Plus reference environment | 7e241bd630a3719a56157a497ce5d08f244784f1 |

## Official loading and evaluation call graph

**VERIFIED FROM PINNED SOURCE**

The complete static path is:

~~~text
lerobot_eval.eval_main
  +-> make_env
  |     -> LiberoEnv.create_envs
  |     -> create_libero_envs
  |     -> LeRobot LiberoEnv
  |     -> hf-libero OffScreenRenderEnv -> TASK_MAPPING -> robosuite/MuJoCo
  |
  +-> make_policy
  |     -> EvalPipelineConfig.__post_init__
  |          -> PreTrainedConfig.from_pretrained(revision omitted for Hub path)
  |     -> SmolVLAPolicy.from_pretrained(config=already loaded config,
  |                                      revision=cfg.pretrained_revision)
  |     -> SmolVLAPolicy.__init__ -> VLAFlowMatching
  |     -> SmolVLMWithExpertModel
  |          -> AutoModelForImageTextToText.from_pretrained(base model name)
  |          -> AutoProcessor.from_pretrained(base model name)
  |     -> hf_hub_download(model.safetensors, revision=checkpoint SHA)
  |
  +-> make_pre_post_processors
  |     -> serialized policy_preprocessor.json plus normalizer state
  |     -> serialized policy_postprocessor.json plus unnormalizer state
  |
  +-> make_env_pre_post_processors
  |     -> LiberoProcessorStep -> identity LIBERO action postprocessor
  |
  +-> eval_policy_all -> eval_policy -> rollout
        -> policy.reset and env.reset(seed)
             -> OffScreenRenderEnv.reset
             -> set_init_state
             -> ten dummy settle actions
        -> preprocess_observation
        -> add task instruction
        -> LiberoProcessorStep
        -> serialized policy preprocessor
        -> SmolVLAPolicy.select_action
             -> _get_action_chunk
             -> prepare_images / prepare_state
             -> VLAFlowMatching.sample_actions
                  -> embed_prefix
                       -> vision_model
                       -> connector
                       -> language embedding + state projection
                  -> SmolVLMWithExpertModel.forward(prefix)
                       -> 32-layer manual loop and prefix KV cache
                  -> euler_integrate(num_steps=10)
                       -> denoise_step
                            -> action/time suffix
                            -> SmolVLMWithExpertModel.forward(expert suffix, cache)
                            -> action_out_proj
             -> crop 32D internal action to 7D
             -> queue first n_action_steps (=1)
        -> serialized action postprocessor
        -> identity LIBERO env postprocessor
        -> CPU NumPy (B, 7)
        -> env.step
~~~

### Revision propagation

**VERIFIED FROM PINNED SOURCE**

The generic APIs are revision-capable: PreTrainedConfig.from_pretrained accepts
revision and would pass it to the config download
(configs/policies.py:171-214), while PreTrainedPolicy.from_pretrained passes
revision to the safetensors download (policies/pretrained.py:168-227).
make_policy supplies cfg.pretrained_revision to policy from_pretrained
(factory.py:334-339).

**VERIFIED FROM PINNED SOURCE**

The official CLI path does not use that config capability.
EvalPipelineConfig.__post_init__ first calls
PreTrainedConfig.from_pretrained(policy_path) without revision
(configs/eval.py:45-54). make_policy later passes this already-created config
object into PreTrainedPolicy.from_pretrained (factory.py:319-339), so the
policy loader skips its own config reload (policies/pretrained.py:188-201).
For a Hub path, current config.json can therefore be combined with weights
addressed by cfg.pretrained_revision.

**VERIFIED FROM PINNED SOURCE**

make_pre_post_processors accepts pretrained_revision and forwards it to both
serialized pipeline loads (factory.py:150-219), but eval_main calls that
factory without pretrained_revision (lerobot_eval.py:776-780). The official
evaluation entry point therefore does not enforce the processor revision even
when the policy revision is present.

**VERIFIED FROM PINNED SOURCE**

SmolVLMWithExpertModel calls both base-model and AutoProcessor from_pretrained
without revision (smolvlm_with_expert.py:90-101), and
TokenizerProcessorStep calls AutoTokenizer.from_pretrained without revision
(tokenizer_processor.py:88-111).

**NOT YET VERIFIED**

The selected base-model SHA is a bootstrap retrieval pin, not recovered
training provenance. The checkpoint does not record historical LeRobot,
LIBERO, or base-VLM SHAs. Before M0, config, weights, serialized processors,
AutoProcessor, and tokenizer must all be made fail-closed to immutable
artifacts by an approved design; using a fully materialized, hash-checked local
checkpoint/base directory is one design candidate. Static audit alone does not
prove the checkpoint loads under Transformers 5.4-5.5 or that DynamicCache
behavior is compatible.

## Observation preprocessing and action postprocessing

### Complete ordered pipeline

**VERIFIED FROM PINNED SOURCE**

| Stage | Input | Operation | Output |
| --- | --- | --- | --- |
| LIBERO wrapper | robosuite observation | Select agentview_image and robot0_eye_in_hand_image; expose nested EEF, gripper, and joint state | pixels/image and pixels/image2 as HWC uint8; nested robot_state |
| Generic conversion | (B,H,W,3) uint8 | torch.from_numpy, HWC to CHW, float32, divide by 255 | (B,3,H,W) float32 in [0,1] |
| Generic conversion | nested NumPy arrays | Recursive conversion to batched tensors | observation.robot_state |
| Eval loop | environment metadata | Insert natural-language task list | task |
| LIBERO processor | both NCHW images | Flip dimensions 2 and 3 | 180-degree-rotated images |
| LIBERO processor | EEF position (B,3), quaternion (B,4), gripper qpos (B,2) | quaternion to axis-angle (B,3), concatenate, float32 | observation.state (B,8) |
| Serialized preprocessor 0 | observation dictionary | rename map; selected artifact has an empty map | same keys |
| Serialized preprocessor 1 | tensors | add batch dimension only when absent | batched tensors |
| Serialized preprocessor 2 | task string/list | ensure trailing newline | task prompts |
| Serialized preprocessor 3 | prompt | right-padding to longest in batch, truncation, maximum 48 | language token IDs and attention mask (B,T), T <= 48 |
| Serialized preprocessor 4 | tensors | move to configured policy device | device tensors |
| Serialized preprocessor 5 | state/images | images IDENTITY; state MEAN_STD using pinned state artifact | normalized model batch |
| SmolVLA internal image path | each (B,3,H,W) image | resize with padding to 512 square; x maps to 2x-1 | (B,3,512,512) in [-1,1] |
| SmolVLA internal state path | (B,8) | zero-pad to max_state_dim 32 | (B,32) |
| Policy output | internal (B,50,32) | crop to checkpoint action dimension | action chunk (B,50,7) |
| Action queue | (B,50,7) | enqueue only first n_action_steps=1 and pop one | (B,7) |
| Serialized postprocessor | (B,7) | move to CPU, then MEAN_STD action unnormalization | CPU (B,7) |
| LIBERO env postprocessor | action transition | empty pipeline | unchanged (B,7) |
| Eval loop | CPU tensor | convert to NumPy | env.step action (B,7) |

Sources: envs/libero.py:145-175 and 287-391; envs/utils.py:68-120;
processor/env_processor.py:26-112; modeling_smolvla.py:195-269 and 334-418;
the pinned policy_preprocessor.json:1-79 and policy_postprocessor.json:1-32.

### Image-size contract

**VERIFIED FROM PINNED SOURCE**

The checkpoint and serialized normalizer metadata declare 256 by 256 images.
The LeRobot environment class constructor also defaults to 256 by 256
(envs/libero.py:119-120), but the registered LiberoEnv configuration defaults
to 360 by 360 and passes that size into the environment
(envs/configs.py:333-334 and 414-426). Visual feature validation checks camera
key-set compatibility, not exact image shapes
(policies/utils.py:226-249). SmolVLA later resizes either size to 512.

**NOT YET VERIFIED**

It is not established that accepting the 360 default is numerically equivalent
to the checkpoint's historical 256 input path. M0 must fail closed unless the
approved configuration explicitly resolves this contract; no silent resize
assumption may be introduced outside the official SmolVLA path.

## SmolVLA architecture and tensors

### Modules

**VERIFIED FROM PINNED SOURCE**

- VLAFlowMatching owns the base SmolVLM-plus-expert wrapper, state_proj
  (32 to 960), action_in_proj (32 to 480), action_out_proj (480 to 32),
  and a two-layer action/timestep MLP (960 to 480 to 480).
- The pinned base configuration has a 32-layer text model with hidden size 960,
  FFN size 2560, 15 query heads, 5 KV heads, and head dimension 64.
- Its vision configuration has image size 512, patch size 16, hidden size 768,
  and 12 attention heads. The connector is called directly after the vision
  model.
- The expert copies the text config, changes hidden size to 480 and FFN size to
  1280, and retains 32 layers. With cross_attn and
  self_attn_every_n_layers=2, even-numbered expert layers perform joint/self
  attention and odd-numbered layers cross-attend to the fixed VLM prefix cache.
- Prefix prefill executes all 32 VLM layers once per action query. Each of ten
  flow steps executes all 32 expert layers: 16 self-attention-mode layers and
  16 cross-attention-mode layers.

### Shape ledger

**VERIFIED FROM PINNED SOURCE**

Rows marked “config-derived” are arithmetic consequences of the pinned
checkpoint/base configuration and the LeRobot assembly code; they have not
been observed in a running Transformers model.

| Boundary | Shape | Basis |
| --- | --- | --- |
| raw image per camera | (B,H,W,3) uint8 | environment wrapper; H/W contract discussed above |
| generic policy image | (B,3,H,W) float32 | preprocess_observation |
| internal resized image | (B,3,512,512) | prepare_images |
| vision patch sequence per image | expected (B,1024,768) | config-derived: (512/16)^2 patches, hidden 768 |
| connector output per image | expected (B,64,960) | config-derived: pixel-shuffle factor 4 per spatial axis, use_resampler=false |
| two-camera visual prefix | expected (B,128,960) | two times 64 tokens |
| language tokens/embeddings | (B,T), then (B,T,960), T <= 48 | serialized tokenizer plus language embedding |
| physical state | (B,8) | LIBERO processor |
| padded/projected state token | (B,32), then (B,1,960) | pad_vector and state_proj |
| complete VLM prefix residual | expected (B,P,960), P=128+T+1 <=177 | two images + language + one state token; no image special tokens |
| prefix KV cache per layer | expected keys/values (B,5,P,64) | 5 KV heads, head dimension 64; DynamicCache stores B,H,L,D |
| initial flow noise/state | (B,50,32), float32 | chunk_size and max_action_dim |
| action/time expert suffix | (B,50,480) | action projection plus timestep MLP |
| expert residual per layer/flow step | (B,50,480) | 32-layer expert |
| predicted flow velocity | (B,50,32), float32 | action_out_proj |
| integrated internal action | (B,50,32) | ten Euler updates |
| cropped action chunk | (B,50,7) | checkpoint action feature |
| action returned by select_action | (B,7) | n_action_steps=1 queue |

**NOT YET VERIFIED**

The connector and vision-block implementation lives in Transformers, whose
exact source commit is not pinned by LeRobot. The 1024-to-64 token calculation
is supported by the pinned configuration but requires a runtime shape probe
before it can be treated as observed behavior. Dtypes after the bfloat16
vision/VLM calls, exact cache classes, and numerical layer counts must also be
confirmed during the later environment preflight.

## Flow noise and explicit-noise seam

**VERIFIED FROM PINNED SOURCE**

sample_noise uses torch.normal with mean 0, standard deviation 1, float32, and
shape (B, chunk_size, max_action_dim). sample_actions samples only when its
noise argument is None. SmolVLAPolicy.predict_action_chunk(batch, noise=xi)
and select_action(batch, noise=xi) both propagate an explicit tensor through
_get_action_chunk to sample_actions.

**VERIFIED FROM PINNED SOURCE**

euler_integrate sets dt=-1/num_steps, evaluates times 1.0, 0.9, ..., 0.1 for
num_steps=10, and applies x_t = x_t + dt*v_t. Prefix features and their KV cache
are computed once per sample_actions call, and the cache is reused for all ten
denoise steps. Suffix K/V appended by self-attention layers are cropped back to
the prefix length after each step.

**VERIFIED FROM PINNED SOURCE**

select_action consumes the supplied noise only when the action queue is empty.
With this checkpoint n_action_steps=1, every environment step empties the
queue and causes a new prefix plus ten-step flow solve. For a checkpoint with a
longer queue, a later call can return a queued action without consuming the
new noise argument.

**PROPOSED DESIGN**

Paired clean/OOD experiments must create one explicit xi tensor of shape
(B,50,32) and pass that identical tensor to both full-chunk queries. The safest
future API is predict_action_chunk, or a controller that verifies and clears
both queues before paired select_action calls. The internal noise tensor,
generator/device, seed, dtype, and digest belong in the run manifest. This seam
already exists in the pin; M0 must not add or use experimental noise plumbing.

## Transformer execution and activation seam

### What actually executes

**VERIFIED FROM PINNED SOURCE**

SmolVLMWithExpertModel.forward does not invoke decoder_layer(hidden_states).
It obtains lists of decoder blocks, loops over layer_idx, calls each
input_layernorm, q/k/v projection, attention implementation, o_proj,
post_attention_layernorm and MLP manually, then applies the two final norm
modules. The prefix call passes inputs_embeds=[prefix,None]; denoise calls pass
inputs_embeds=[None,suffix].

**VERIFIED FROM PINNED SOURCE**

Consequently ordinary forward hooks attached to complete VLM or expert decoder
blocks will not fire. Hooks on the complete SmolVLMWithExpertModel or
VLAFlowMatching module are also bypassed where callers invoke their .forward
or .sample_actions methods directly. Child modules that are called normally,
including vision_model, connector, layer norms, projections, MLPs and final
norms, do traverse nn.Module.__call__ and are statically hookable.

**VERIFIED FROM PINNED SOURCE**

The attention kernel returned by get_attention_interface is the plain Python
method eager_attention_forward, not an nn.Module, so it has no
register_forward_hook seam (smolvlm_with_expert.py:511-516). The eval loop also
calls policy.select_action directly rather than policy(...), so an ordinary
hook on the outer SmolVLAPolicy module does not fire.

**NOT YET VERIFIED**

No hook counter probe has run. Vision encoder internals are owned by the
unpinned Transformers implementation, and torch.compile or library changes can
alter hook behavior. Runtime counters and a disabled-controller no-op
equivalence test are required before any intervention milestone.

### Layer-boundary intervention

**PROPOSED DESIGN**

Use forward pre-hooks on each next-layer input_layernorm to observe the complete
incoming residual tensor, and on each final norm to cover the last residual
boundary. Do not hook whole decoder blocks.

The ShiftVLA controller must tag every event with:

- phase: prefix or denoise;
- stream: vlm or expert;
- layer: 0 through 31, with final as a distinct boundary;
- flow_step: null for prefix, otherwise 0 through 9;
- branch: clean or OOD;
- trajectory/state ID and explicit-noise ID.

For layer k, the input_layernorm pre-hook sees the residual produced by layer
k-1; for layer 0 it sees the assembled prefix or suffix embedding. Returning a
replacement argument from that hook changes only the layer-normalized branch:
the manual loop's later skip addition still references its original
hidden_states tensor. A full-residual exchange is therefore possible through
this hook only if a no-grad in-place copy into the hook input is proven to
mutate the same tensor alias used by the skip path. That aliasing and no-op
behavior are NOT YET VERIFIED.

The clean/OOD exchange must preserve tensor shape, dtype, device, padding mask,
phase and flow step. The final-norm pre-hook can replace its argument directly
because there is no parallel skip use after that boundary. A runtime controller
disabled by default must be demonstrably identical to unmodified inference.
If the pinned probe shows that safe in-place replacement is unreliable, the
later milestone must propose a minimal, disabled-by-default explicit callback
at the residual assignment itself. It must not mislabel a normalized-branch
patch as complete residual exchange. No hook or callback is implemented here.

## LIBERO reset and state APIs

### Existing APIs

**VERIFIED FROM PINNED SOURCE**

LeRobot LiberoEnv.reset seeds the backend, calls reset, applies one benchmark
initial state through set_init_state, advances init_state_id by the vector
stride, and then takes the configurable num_steps_wait (default ten) dummy
actions [0,0,0,0,0,0,-1] before returning the formatted observation
(envs/libero.py:90-92 and 339-364). Absolute/relative controller mode is set
only after those settle steps. This is benchmark initialization, not arbitrary
mid-trajectory replay.

**VERIFIED FROM PINNED SOURCE**

hf-libero ControlEnv.get_sim_state calls robosuite get_state().flatten, while
set_state only calls set_state_from_flattened
(env_wrapper.py:115-128). The separate set_init_state path delegates to
regenerate_obs_from_state, which calls set_state, forward, post-processing and
forced observable refresh (env_wrapper.py:136-145). Pinned robosuite
MjSimState contains only time, qpos and qvel, with flattened length 1+nq+nv
and an assertion that na is zero (binding_utils.py:213-241). It omits
activation, history, warmstart, control, applied forces, equality activation,
mocap, userdata and plugin state.

**VERIFIED FROM PINNED SOURCE**

MuJoCo 3.7.0 defines mjSTATE_INTEGRATION as time, qpos, qvel, activation,
history, plugin state, controls, applied forces, equality state, mocap,
userdata and warmstart (mjdata.h:27-52). Its Python bindings expose
mj_stateSize, mj_getState and mj_setState with length validation
(functions.cc:325-359). MuJoCo documents the integration state as the complete
input to forward dynamics for ordinary cases.

**NOT YET VERIFIED**

Neither the installed MuJoCo Python version nor the live robosuite wrapper has
been probed. Controller-private state, observable buffers, RNGs, sleeping
islands, plugins and exact replay across separately compiled camera models
remain unresolved. The vector worker command protocol does not expose a public
arbitrary full-state setter.

### Exact mid-trajectory replay seam

**PROPOSED DESIGN**

A future ShiftVLA-owned wrapper should capture only after a completed
high-level LiberoEnv.step, inside the owning worker:

1. Record the full MJCF/XML hash, MuJoCo version,
   mjSTATE_INTEGRATION signature and mj_stateSize.
2. Build a machine-checkable compatibility signature containing every
   integration-state field size; nq/nv/na, nhistory, npluginstate, neq, nmocap
   and nuserdata; joint/body/geom/camera/actuator names and ordering; dynamics
   options; and plugin type/layout.
3. Read mjSTATE_INTEGRATION from the live model/data, not robosuite's flattened
   MjSimState.
4. Record robosuite episode counters, controller/integrator state, observable
   buffers or refresh contract, Python/NumPy/MuJoCo RNG state, task,
   instruction, episode/init-state ID and the last high-level action.
5. For same-model restore, require the exact model hash. For clean/OOD transfer,
   record both full hashes and require equal state layout/object/actuator
   ordering plus an audited MJCF diff restricted to the allowed camera
   pos/quat fields; exact XML hash equality would be wrong because the camera
   perturbation intentionally changes XML.
6. Reject sleeping-island or incompatible plugin cases before capture. For
   ordinary non-sleeping compatible models, call mj_setState with
   mjSTATE_INTEGRATION, restore wrapper/controller
   metadata, call mj_forward, then reproduce the forced observable refresh and
   LeRobot _format_raw_obs path.
7. Do not call reset, set_init_state, or dummy settling during restoration.
8. Assert source/destination state-vector equality before observation,
   camera-independent derived-dynamics equality after mj_forward, and identical
   next state/reward after one shared high-level action.

If the eventual MuJoCo runtime lacks this full state API, if the two compiled
models have incompatible state layouts, or if private controller/RNG state
cannot be made exact, M1 is blocked. It must not fall back to time/qpos/qvel or
weaken the matched-state scientific definition. Same-model full-mjData copying
may be evaluated separately for sleeping states, but it does not solve
cross-model clean/OOD transfer and is outside this proposed seam.

## LIBERO-Plus camera representation

### Static representation

**VERIFIED FROM PINNED SOURCE**

Camera variants are encoded in BDDL filenames:

~~~text
<base>_view_<horizon-horizontal>_<vertical>_<scale-percent>
       _<endpoint-rotation>_<endpoint-vertical>
       _initstate_<id>[_noise_<id>].bddl
~~~

ControlEnv splits the suffix, divides the integer scale field by 100, and
passes the five camera values to TASK_MAPPING. A nonzero initstate changes the
robot class name; optional noise applies an image corruption to agentview after
reset. Camera-only comparisons must therefore use initstate_0 and no noise
suffix.

**VERIFIED FROM PINNED SOURCE**

During BDDLBaseDomain._load_model, _setup_camera is called on the MJCF arena
before ManipulationTask is composed. The tabletop implementation rotates the
baseline agentview position/quaternion around Y and Z axes, optionally scales
the position vector around pivot (0,0,0.8), applies endpoint orientation
rotations, and calls Arena.set_camera(agentview, pos, quat). robosuite then
serializes that XML and creates MjModel from the XML string. Camera
perturbations are therefore compile-time model parameters in this pin.

**VERIFIED FROM PINNED SOURCE**

scale is a camera-distance multiplier around a pivot, not field of view. The
camera call supplies pos and quat and does not write fovy or camera_attribs.
Although the Plus README describes FOV changes, FOV perturbation is not
implemented by this pinned camera path.

**VERIFIED FROM PINNED SOURCE**

LeRobot LiberoPlusEnv only changes the suite default and
is_libero_plus flag (envs/configs.py:729-750). Its initial-state loader strips
the virtual view suffix and loads the base .pruned_init
(envs/libero.py:60-87). The Plus benchmark hard-codes suite counts
2402+2518+2591+2519+90=10120, while the README and classification JSON describe
10030 perturbation tasks across the four primary suites. The extra 90 aligns
arithmetically with libero_90, but intended evaluation coverage and asset
completeness are not runtime-verified.

### Matched camera seam

**PROPOSED DESIGN**

Construct two independent, already-compiled environments from the same pinned
LIBERO-Plus source and asset set: one base/zero-view clean task and one
view-perturbed counterpart with initstate_0. Verify their MJCF difference is
limited to approved agentview camera pos/quat, their dynamic state layouts are
identical, and their task instruction and base initial-state artifact hashes
match. Restore one captured integration state into both, refresh observations,
and require proprioception and the wrist image to remain equal while only the
agentview observation changes. Never pair vanilla hf-libero against Plus,
mutate a camera in one already-compiled environment, or include language,
light, table, object, noise, or nonzero-initstate suffixes in the camera-only
pilot.

## Architecture-aware component grouping

**PROPOSED DESIGN**

The pinned layer count is 32, so the fixed eight candidate groups are:

| Group | Components | Layer boundary |
| --- | --- | --- |
| 1 | vision encoder | vision_model output for each camera |
| 2 | connector / visual-token projector | connector output before prefix concatenation |
| 3 | early VLM blocks | VLM layers 0-15 |
| 4 | late VLM blocks | VLM layers 16-31 |
| 5 | state, action and timestep conditioning interface | state_proj; action_in_proj; sinusoidal time embedding; action_time_mlp_in/out; prefix/suffix assembly |
| 6 | early action-expert blocks | expert layers 0-15 across each denoise step |
| 7 | late action-expert blocks | expert layers 16-31 across each denoise step |
| 8 | final norm and action output projection | expert final norm, action_out_proj and 32D-to-7D crop |

Groups 3, 4, 6 and 7 use the layer-norm boundary seam above, not whole-block
hooks. Group 5 must distinguish the one-time prefix state token from the
per-flow-step action/timestep suffix. Component discovery and held-out
validation remain trajectory-split as required by the research specification.

## Known risks and unresolved gates

**NOT YET VERIFIED**

1. Checkpoint weights, normalization payload and base-model weights are LFS
   pointers only; no load or numerical inference has run.
2. Policy config, serialized processors, AutoProcessor, base model and
   tokenizer revision propagation is not fail-closed in the official eval
   path; only the policy-weight API receives cfg.pretrained_revision after the
   CLI has already loaded a config.
3. LeRobot v0.6.1 requires Python >=3.12, torch >=2.7,<2.12, NumPy >=2,<2.3,
   Gymnasium >=1.1.1 and Transformers >=5.4,<5.6. The observed shell has Python
   3.13.13 and NumPy 2.4.6 but no torch, Transformers, Gymnasium, MuJoCo,
   hf-libero or robosuite installed; its NumPy is already outside LeRobot's
   declared range. Any separately reported Python 3.10 vendor environment was
   not inspected here and would fail LeRobot's Python floor if selected.
4. The base metadata records older Transformers configuration versions; exact
   Transformers 5.x model construction and DynamicCache behavior are untested.
5. The registered LIBERO environment's 360 image default differs from the
   checkpoint's 256 metadata.
6. canonical LIBERO, hf-libero and LIBERO-Plus all expose import libero.
   Vanilla and Plus must use isolated environments with recorded import paths.
7. hf-libero requires robosuite 1.4.0; LeRobot's Plus Dockerfile installs
   robosuite 1.4.1. Plus's legacy requirements also conflict with LeRobot's
   NumPy, Transformers and Gymnasium stack.
8. LeRobot's lock resolves MuJoCo 3.8.1, while this audit pins MuJoCo 3.7.0
   because the Plus reference Dockerfile selects it. The eventual runtime
   version remains an explicit decision.
9. The Plus reference Dockerfile uses a mutable base image, a short source SHA,
   and an asset download without revision. It is evidence, not a complete
   reproducible environment.
10. Plus assets are not materialized; 10030 versus 10120 task scope, BDDL/state
    coverage and task-ID-to-camera mapping remain unvalidated.
11. EGL reset/render, HCU BF16, checkpoint load, one action chunk, hook counters,
    state replay and repeated-run numerical consistency are all untested.
12. Exact historical training provenance for the checkpoint is unavailable;
    the current base-model pin must not be described as recovered provenance.

## M0 baseline

**PROPOSED DESIGN — NOT IMPLEMENTED; WAITING FOR HUMAN APPROVAL**

M0 should add only ShiftVLA-owned launcher/config/test code after approval. It
should call pinned LeRobot's make_env, make_policy,
make_pre_post_processors and make_env_pre_post_processors, then use the
official eval_policy_all -> eval_policy -> rollout stack rather than copying
the rollout or preprocessing chain.

The proposed episode trace wrapper is passive around reset and step. It records
suite, task ID/name, seed, selected initial-state ID, success, reward and
rollout length, and must pass observation, action, reward, termination,
truncation and info through unchanged. A transparency test must compare wrapped
and unwrapped calls.

The public reset result contains only is_success; it does not expose the
selected initial-state ID, and the inner LiberoEnv increments its private
init_state_id during reset. A passive outer vector wrapper must not pretend it
observed that value. M0 must either (a) derive the selected ID from the pinned
episode_index + reset_count*n_envs rule and verify the derivation against an
in-worker read-only probe, or (b) obtain approval for a passive in-worker trace
wrapper that snapshots the ID before delegating reset. If neither can be
verified without changing reset behavior, exact init-state recording is an M0
blocker.

The policy is frozen with eval and torch.inference_mode. M0 creates no
optimizer, never imports a training entry point, enables no PEFT/LoRA, and
contains no state replay, explicit paired-noise path, hook or activation
controller. HCU selection remains device=cuda through the vendor-compatible
PyTorch API; no physical device is hard-coded.

M0's vanilla runtime should contain only pinned hf-libero. Canonical LIBERO is
provenance source, not a co-installed package, and LIBERO-Plus remains in a
separate future environment so that the three libero namespaces cannot shadow
one another.

Proposed configurations:

- smoke: libero_spatial task 0, seed 2027, one episode, batch size 1, serial,
  one video, pinned official episode length/camera mapping/action queue;
- full: the four standard ten-task suites, ten episodes per task, batch size 1,
  serial, with explicit task IDs in configuration rather than source code.

M0 output contract:

- run_manifest.json with the exact checkpoint identifier and revision,
  ShiftVLA/dependency/base-model revisions and hashes, exact command/config,
  Python/package/MuJoCo/rendering versions, torch.version.cuda,
  torch.version.hip, DTK, actual HCU name, seed,
  action_generation_noise_seed, flow-noise RNG provenance,
  task IDs, trajectory IDs, init-state IDs, perturbation_settings explicitly
  set to a no-shift baseline, timing, status and failure;
- episodes.jsonl with suite, task ID/name, episode, stable trajectory_id,
  init-state ID, seed, success, rollout length, reward, perturbation settings
  and video path;
- raw eval_info.json, logs and videos.

Proposed M0 tests cover manifest/pin schema validation, immutable checkpoint
revision propagation, exact official processor ordering, trace-wrapper
transparency, frozen/eval/inference mode with no optimizer or LoRA, and a
failure-path manifest that is written before a fail-closed exit. These are
implementation acceptance tests, not permission to implement them in this
phase.

Before implementation, the approved M0 design must resolve fail-closed
checkpoint config, checkpoint weights, serialized processors, base model, and
tokenizer loading; the 256-versus-360 image contract; the exact vanilla runtime
lock; and the hf-libero import path. Preflight must
then pass device discovery, BF16, checkpoint load, one action chunk, and
MUJOCO_GL=egl reset/render in that order. EGL failure stops the run; no OSMesa
or renderer fallback is permitted.

Acceptance requires two runs of the same smoke configuration to agree, within
an approved tolerance, on task, init-state, initial processed observation,
action sequence, rollout length and success. A failure does not authorize
pulling M3 explicit-noise intervention into M0.

The older engineering-plan template names lerobot/smolvla_libero, while the
current approved bootstrap plan deliberately selects
HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102
for its two-camera/8D contract. The current pin controls this proposed M0
design; changing back to the rejected candidate requires a new human decision.

## Bootstrap stop condition

**VERIFIED FROM PINNED SOURCE**

This document completes static bootstrap reconnaissance only.

**PROPOSED DESIGN**

Stop here. Do not implement M0 or any later milestone until this architecture
map and its unresolved gates receive human approval.
