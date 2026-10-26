# VoiceStudio: CosyVoice Engine

CosyVoice is an optional multilingual TTS backend for zero-shot voice cloning
and instructed speech. A one-click install runs it in its own
environment and process; an existing source installation keeps
running in-process.

## Downloaded weights and an available engine are different states

The Model Catalogue tracks model weights separately from engine runtime
availability. A downloaded
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` cache means the model files reached the
machine. It does not prove that the VoiceStudio backend can import and run
CosyVoice.

For an existing source installation, the same Python interpreter that runs
the VoiceStudio backend must import:

```python
from cosyvoice.cli.cosyvoice import AutoModel
```

It then loads the directory named by `OMNIVOICE_COSYVOICE_MODEL`, or defaults
to `pretrained_models/Fun-CosyVoice3-0.5B`. That path must resolve to the usable
model directory, not only the parent Hugging Face cache directory.

## One-click install

Click **Install** in **Model Catalogue → Engines → CosyVoice**. VoiceStudio
gives CosyVoice its own folder under the data directory and its own Python
3.10 environment, and runs it there in a separate process. The install:

- clones a reviewed CosyVoice commit and the Matcha-TTS code it depends on;
- downloads the CosyVoice 3 weights (about 5.4 GB).

It differs from upstream's own setup:

- **PyTorch 2.7.0.** The CUDA 12.8 build on an NVIDIA GPU, the CPU build on
  other Windows and Linux machines, the regular build on Apple Silicon.
  Upstream's 2.3.1 exists only for CUDA 12.1 and cannot run on RTX 50-series
  GPUs.
- **Leaner dependencies.** No TensorRT, DeepSpeed or GPU onnxruntime, and no
  third-party package index. Upstream uses them for extra speed on Linux;
  synthesis works without them. SoX is not needed.
- **PyWORLD per platform.** On Windows the install uses the published
  `pyworld==0.3.5` wheel, so MSVC Build Tools are not required. Linux and
  macOS build the pinned PyWORLD source, which needs a C++ toolchain —
  Xcode Command Line Tools on macOS, the distribution's C++ build tools on
  Linux — and the install checks for one before downloading anything, so a
  missing toolchain fails with the fix-it instruction instead of a compiler
  traceback mid-download.
- **Patched dependencies.** Where upstream pins a release with a published
  security advisory (diffusers, hydra-core, lightning, modelscope, onnx,
  protobuf, transformers), the install uses the fixed release. That set was
  installed on Windows and passes the install's import check.
- **No text normaliser.** Upstream's (wetext) downloads its data from
  ModelScope on every model load, and ModelScope rate-limits those
  downloads, so a normaliser could half-download and fail silently. The
  one-click install leaves it out, and CosyVoice reads text as written:
  spell out numbers, dates and symbols where the pronunciation matters.
- **Only the weights CosyVoice 3 loads.** Not the RL and TensorRT variants
  that share its repository.

Nothing it installs touches VoiceStudio itself or any other engine, and
**Uninstall** in the same row removes only that folder. An existing source
installation keeps working as it is. The button is not offered on Intel
Macs, where PyTorch 2.7.0 has no build.

CosyVoice 3 has no built-in voices. With no reference clip, it speaks in the
voice of upstream's own sample prompt.

## Source builds and existing installations

The upstream CosyVoice project recommends its own Python 3.10 Conda environment
and SoX. The one-click install above gives CosyVoice that separate
environment. Installing upstream dependency pins into VoiceStudio's
shared backend environment can also conflict with other engines.

Existing source installations remain usable when the VoiceStudio backend's
interpreter can already import `cosyvoice.cli.cosyvoice.AutoModel`. Keep a
working installation in place. Before starting VoiceStudio, set
`OMNIVOICE_COSYVOICE_MODEL` to its usable model directory through the source
checkout's environment or project `.env` file.

A clone you keep outside VoiceStudio can also run as its own sidecar: point
`OMNIVOICE_COSYVOICE_DIR` at the checkout and keep a `.venv` directory inside
it with CosyVoice's dependencies installed. VoiceStudio spawns that
environment directly, so the dependencies cannot live in a system or Conda
interpreter — only inside the clone's `.venv`. When the folder is set but has
no such environment, the engine row says exactly that and points at the
one-click install above, not at the upstream manual steps. The app never
provisions or rewrites a user-managed clone; a complete managed install and
your own `.venv` keep working side by side.

For upstream setup details, read the
[official CosyVoice installation guide](https://github.com/QwenAudio/CosyVoice#install).
Those steps create a standalone CosyVoice environment. Prefer the managed
one-click installer above for the Electron app.

## Diagnose an unavailable engine

Include these facts in a support question or bug report:

1. The exact VoiceStudio version.
2. The full reason under **Model Catalogue > Engines > CosyVoice** after
   selecting **Re-check**.
3. The CosyVoice lines from **Settings > Logs > Backend** immediately after
   that check.
These facts distinguish a downloaded-model state from a missing Python runtime
or incorrect model directory. Do not delete a model cache or reinstall
dependencies until the log identifies which state failed. SoX is not part of
CosyVoice's runtime and never the cause of an unavailable engine here.

The public report that exposed the misleading installed state is
[Discussion 1631](https://github.com/debpalash/VoiceStudio/discussions/1631).


On hosts without CUDA, the managed sidecar normalizes the LLM, flow and vocoder
weights to float32 to match upstream CPU inputs. CUDA keeps its selected
precision. This prevents the CPU Float/BFloat16 matrix mismatch; it does not
establish that every reported installation or speech-quality problem is fixed.

### Repairing missing runtime imports

The managed recipe includes gdown, wget and pyarrow, and builds pinned PyWORLD
source that no longer needs `pkg_resources`. Its final check imports the dataset
processor and Matcha utilities as well as AutoModel. Older completion markers
are treated as needing repair; retry Install to reuse the existing model weights.
User-managed environments are not rewritten. Do not downgrade setuptools merely
to restore `pkg_resources`.

### Speech context with newer Transformers

The managed sidecar loads Qwen in float32 before applying the trained checkpoint,
then preserves upstream's explicit precision choices. It also includes cached
prompt tokens in incremental attention masks: a one-token mask in newer
Transformers otherwise hides the text and voice context and produces unrelated
or garbled speech. Existing Transformers 4 environments and legacy cache tuples
remain supported; full masks are left intact.

A real CosyVoice 3 CPU check with Transformers 5.10.1 now reproduces the complete
89-word English input, verified with an independent local speech recognizer.
A tiny randomly initialized Qwen regression checks cached versus full-context
decoding without downloading any model in CI. This does not certify every
language, voice reference, or GPU configuration.
