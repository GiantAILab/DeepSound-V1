<!-- # DeepSound-V1
Official code for DeepSound-V1 -->


<div align="center">
<p align="center">
  <h2>DeepSound-V1</h2>
  <!-- <a href="https://arxiv.org/abs/2412.15322">Paper</a> | <a href="https://hkchengrex.github.io/MMAudio">Webpage</a> | <a href="https://huggingface.co/hkchengrex/MMAudio/tree/main">Models</a> | <a href="https://huggingface.co/spaces/hkchengrex/MMAudio"> Huggingface Demo</a> | <a href="https://colab.research.google.com/drive/1TAaXCY2-kPk4xE4PwKB3EqFbSnkUuzZ8?usp=sharing">Colab Demo</a> | <a href="https://replicate.com/zsxkib/mmaudio">Replicate Demo</a> -->
  <a href="https://github.com/lym0302/DeepSound-V1">Paper</a> | <a href="https://github.com/lym0302/DeepSound-V1">Webpage</a> | <a href="https://github.com/lym0302/DeepSound-V1"> Huggingface Demo</a>
</p>
</div>

## [DeepSound-V1: Start to Think Step-by-Step in the Audio Generation from Videos](https://github.com/lym0302/DeepSound-V1)

<!-- [Ho Kei Cheng](https://hkchengrex.github.io/), [Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ), [Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ), [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ), [Alexander Schwing](https://www.alexander-schwing.de/), [Yuki Mitsufuji](https://www.yukimitsufuji.com/) -->

<!-- University of Illinois Urbana-Champaign, Sony AI, and Sony Group Corporation -->

<!-- ICCV 2025 -->

## Highlight

DeepSound-V1 is a framework enabling audio generation from videos towards initial step-by-step thinking without extra annotations based on the internal chain-of-thought (CoT) of Multi-modal large language model(MLLM).

<!-- ## Results

(All audio from our algorithm MMAudio)

Videos from Sora:

https://github.com/user-attachments/assets/82afd192-0cee-48a1-86ca-bd39b8c8f330

Videos from Veo 2:

https://github.com/user-attachments/assets/8a11419e-fee2-46e0-9e67-dfb03c48d00e

Videos from MovieGen/Hunyuan Video/VGGSound:

https://github.com/user-attachments/assets/29230d4e-21c1-4cf8-a221-c28f2af6d0ca

For more results, visit https://hkchengrex.com/MMAudio/video_main.html. -->


## Installation
```bash
conda create -n deepsound-v1 python=3.10.16 -y
conda activate deepsound-v1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
cd requirements
pip install -e .
pip install -r reqirments.txt
```


<!-- We have only tested this on Ubuntu.

### Prerequisites

We recommend using a [miniforge](https://github.com/conda-forge/miniforge) environment.

- Python 3.9+
- PyTorch **2.5.1+** and corresponding torchvision/torchaudio (pick your CUDA version https://pytorch.org/, pip install recommended)
<!-- - ffmpeg<7 ([this is required by torchaudio](https://pytorch.org/audio/master/installation.html#optional-dependencies), you can install it in a miniforge environment with `conda install -c conda-forge 'ffmpeg<7'`) -->

<!-- **1. Install prerequisite if not yet met:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

(Or any other CUDA versions that your GPUs/driver support) -->

<!-- ```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg) -->

<!-- **2. Clone our repository:**

```bash
git clone https://github.com/lym0302/DeepSound-V1.git
```

**3. Install with pip (install pytorch first before attempting this!):**

```bash
cd DeepSound-V1
pip install -e .
```

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip) --> 


<!-- The models will be downloaded automatically when you run the demo script. MD5 checksums are provided in `mmaudio/utils/download_utils.py`.
The models are also available at https://huggingface.co/hkchengrex/MMAudio/tree/main
See [MODELS.md](docs/MODELS.md) for more details. -->

## Demo

### Pretrained models
See [MODELS.md](docs/MODELS.md).

### Command-line interface

With `demo.py`

```bash
python demo.py \
    --video=<path to video> \
    --prompt "your prompt" \
    --step1 "mmaudio-l" \
    --step2 "cot" \
    --step3 "bsroformer" \
    --step4 "neg"
```

All training parameters are [here]().

<!-- The output (audio in `.wav` format, and video in `.mp4` format) will be saved in `./output`.
See the file for more options.
Simply omit the `--video` option for text-to-audio synthesis.
The default output (and training) duration is 8 seconds. Longer/shorter durations could also work, but a large deviation from the training duration may result in a lower quality. -->

<!-- ### Gradio interface

Supports video-to-audio and text-to-audio synthesis.
You can also try experimental image-to-audio synthesis which duplicates the input image to a video for processing. This might be interesting to some but it is not something MMAudio has been trained for.
Use [port forwarding](https://unix.stackexchange.com/questions/115897/whats-ssh-port-forwarding-and-whats-the-difference-between-ssh-local-and-remot) (e.g., `ssh -L 7860:localhost:7860 server`) if necessary. The default port is `7860` which you can specify with `--port`.

```bash
python gradio_demo.py
``` -->



## Evaluation

See [EVAL.md](docs/EVAL.md).


## Citation

<!-- ```bibtex
@inproceedings{cheng2025taming,
  title={Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis},
  author={Cheng, Ho Kei and Ishii, Masato and Hayakawa, Akio and Shibuya, Takashi and Schwing, Alexander and Mitsufuji, Yuki},
  booktitle={CVPR},
  year={2025}
}
``` -->

## Relevant Repositories

- [av-benchmark](https://github.com/hkchengrex/av-benchmark) for benchmarking results.


## Acknowledgement

Many thanks to:
- [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) 
- [MMAudio](https://github.com/hkchengrex/MMAudio) 
- [FoleyCrafter](https://github.com/open-mmlab/FoleyCrafter)
- [BS-RoFormer](https://github.com/ZFTurbo/Music-Source-Separation-Training) 