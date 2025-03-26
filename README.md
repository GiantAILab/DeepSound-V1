<!-- # DeepSound-V1
Official code for DeepSound-V1 -->


<div align="center">
<p align="center">
  <h2>DeepSound-V1</h2>
  <a href="https://github.com/lym0302/DeepSound-V1">Paper</a> |  <a href="https://huggingface.co/spaces/lym0302/DeepSound-V1"> Huggingface Demo</a>
</p>
</div>

## [DeepSound-V1: Start to Think Step-by-Step in the Audio Generation from Videos](https://github.com/lym0302/DeepSound-V1)


## Highlight

DeepSound-V1 is a framework enabling audio generation from videos towards initial step-by-step thinking without extra annotations based on the internal chain-of-thought (CoT) of Multi-modal large language model(MLLM).


## Installation
```bash
conda create -n deepsound-v1 python=3.10.16 -y
conda activate deepsound-v1
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu120
pip install flash-attn==2.5.8 --no-build-isolation
pip install -e .
pip install -r reqirments.txt
```


## Demo

### Pretrained models
See [MODELS.md](docs/MODELS.md).

### Quick Start
```bash
# coding = utf-8

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
mmaudio_path = os.path.join(project_root, 'third_party', 'MMAudio')
sys.path.append(mmaudio_path)

import argparse

from pipeline.pipeline import Pipeline
from third_party.MMAudio.mmaudio.eval_utils import setup_eval_logging
import os
from moviepy.editor import AudioFileClip, VideoFileClip
import torch
from pathlib import Path
import subprocess
import time

@torch.inference_mode()
def init_pipeline(step0_model_dir='pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT',
                  step1_mode='mmaudio_medium_44k',
                  step2_model_dir='cot',
                  step2_mode='pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT',
                  step3_mode='bs_roformer',):
    st = time.time()
    pipeline = Pipeline(
        step0_model_dir=step0_model_dir, 
        step1_mode=step1_mode, 
        step2_model_dir=step2_model_dir,
        step2_mode=step2_mode,
        step3_mode=step3_mode,
    )
    et = time.time()
    print(f"Initialize models time: {et - st:.2f} s.")
    return pipeline


@torch.inference_mode()
def video_to_audio(pipeline, video_input, output_dir, mode='s4', postp_mode='neg', prompt='', negative_prompt='', duration=10):
    st_infer = time.time()
    step_results = pipeline.run(video_input=video_input, 
                                output_dir=output_dir,
                                mode=mode,
                                postp_mode=postp_mode,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                duration=duration)
    
    temp_final_audio_path = step_results["temp_final_audio_path"]
    temp_final_video_path = step_results["temp_final_video_path"]
    final_audio_path = str(Path(output_dir).expanduser() / f'{Path(video_input).expanduser().stem}.wav')
    final_video_path = str(Path(output_dir).expanduser() / f'{Path(video_input).expanduser().stem}.mp4')

    if temp_final_audio_path is not None:
        subprocess.run(['cp', str(temp_final_audio_path), final_audio_path], check=True)
        step_results["final_audio_path"] = final_audio_path
        if args.skip_final_video:
            step_results["final_video_path"] = None
        else:
            if temp_final_video_path is not None:
                subprocess.run(['cp', str(temp_final_video_path), final_video_path], check=True)
            else:
                audio = AudioFileClip(final_audio_path)
                video = VideoFileClip(video_input)
                duration = min(audio.duration, video.duration)
                audio = audio.subclip(0, duration)
                video.audio = audio
                video = video.subclip(0, duration)
                video.write_videofile(final_video_path)
            step_results["final_video_path"] = final_video_path

    
    et_infer = time.time()
    print(f"Inference time: {et_infer - st_infer:.2f} s.")
    return step_results


@torch.inference_mode()
def main():
    setup_eval_logging()
    video_input = "aa.mp4"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = init_pipeline()
    step_results = video_to_audio(pipeline, video_input, output_dir)

    print("step_results: ", step_results)
    print(f"final_audio_path: {step_results['final_audio_path']}")
    print(f"final_video_path: {step_results['final_video_path']}")


if __name__ == '__main__':
    main()
```

The output (audio in `.wav` format, and video in `.mp4` format) will be saved in `./output`.


With `demo.py`

```bash
python demo.py -i <video_path>
```

Other parameters are [here]().




### Gradio interface

```bash
python gradio_demo.py
```



## Evaluation
Refer [av-benchmark](https://github.com/hkchengrex/av-benchmark) for benchmarking results.
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