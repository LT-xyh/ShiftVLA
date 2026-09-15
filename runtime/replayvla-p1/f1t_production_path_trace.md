# F1T production-path static trace

Authority: `docs/replayvla-p1/06_d3_final_runtime_pivot.md` at commit
`411320d01cdb983b81d1f04a9b378ebe4f634cfe`.

## CPU observation to request bundle

1. The official environment observation arrives in the CPU process as the
   LIBERO/vector-environment observation mapping; image/state leaves may be
   `ndarray` values at this point.
2. `scripts.dcu_preflight._prepare_features` invokes the official
   `preprocess_observation`, adds the task, then invokes the official environment
   and policy preprocessors. Its declared result is `dict[str, torch.Tensor]`.
3. `scripts.dcu_preflight._feature_bundle` validates that every production
   feature is already a `torch.Tensor` and selects the exact feature schema.
   Explicit flow noise is generated on the CPU as a `torch.Tensor` for the
   predict path.
4. `scripts.dcu_preflight._save_bundle` calls
   `scripts.dcu_worker.save_tensor_bundle`; tensors are detached, made
   contiguous, moved to CPU, and serialized as `.safetensors`. The control
   channel carries only JSON metadata and the local file path/hash.

Boundary types:

- environment output: mapping containing `ndarray`/scalar values in CPU process;
- after official CPU preprocessing: `torch.Tensor` mapping;
- IPC data: `.safetensors` bytes;
- IPC control: JSON metadata/path/hash.

## DCU request to policy input

1. `scripts.dcu_model_worker._request_bundle` calls
   `scripts.dcu_worker.load_tensor_bundle`, which uses
   `safetensors.torch.load_file(..., device="cpu")` and returns CPU
   `torch.Tensor` values.
2. `scripts.dcu_model_worker._move` passes each tensor to the device adapter for
   logical `cuda:0` under the installed HIP/DCU build.
3. `predict_action_chunk` and `select_action` pass those device tensors directly
   to the official policy API. No `.numpy()`, `torch.from_numpy`, or
   `torch.as_tensor(ndarray)` call exists on this DCU-side request path.

Boundary types: `.safetensors` -> CPU `torch.Tensor` -> K100 device
`torch.Tensor`.

## DCU action to CPU application

1. The policy returns a DCU `torch.Tensor` (`action_chunk` or `action`).
2. `scripts.dcu_model_worker` validates it and calls
   `save_tensor_bundle`; serialization explicitly moves it to CPU and writes a
   response `.safetensors` file.
3. `scripts.dcu_worker.DCUWorkerClient` loads the response through
   `load_tensor_bundle` in the CPU process and returns a CPU `torch.Tensor`.
4. `scripts.dcu_preflight._postprocess_first_action` passes that tensor through
   the official CPU postprocessor and environment postprocessor before the
   environment action boundary. Any conversion needed by the environment is
   therefore owned by the CPU runtime.

Boundary types: K100 device `torch.Tensor` -> response `.safetensors` -> CPU
`torch.Tensor` -> official CPU postprocessing -> environment action value.

## Conclusion

DCU-side NumPy conversion required: NO

The production path may convert environment-origin NumPy values during CPU-side
official preprocessing or application, but the cross-process and DCU policy
boundary is Torch/safetensors throughout. The failed DCU NumPy C-API bridge is
not an unavoidable production edge in the traced path.
