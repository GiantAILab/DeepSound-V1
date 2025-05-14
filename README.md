<!-- # DeepSound-V1
Official code for DeepSound-V1 -->


<div align="center">
<p align="center">
  <h2>DeepSound-V1</h2>
  <a href="https://arxiv.org/abs/2503.22208">Paper</a> | <a href="https://giantailab.github.io/DeepSound-V1/">Webpage</a>| <a href="https://huggingface.co/spaces/lym0302/DeepSound-V1"> Huggingface Demo</a>| <a href="https://huggingface.co/spaces/lym0302/DeepSound-V1"> Models </a> |<a href="https://huggingface.co/datasets/lym0302/DeepSound-V1"> Dataset</a> | <a href="https://colab.research.google.com/drive/1FcsW9lyu3B0SpmZXj3FYtmhqesXdPU9G?usp=sharing">Colab Demo</a> | <a href="https://github.com/lym0302/DeepSound-V1">Replicate Demo</a>
</p>
</div>

## [DeepSound-V1: Start to Think Step-by-Step in the Audio Generation from Videos](https://github.com/lym0302/DeepSound-V1)

[Yunming Liang](https://scholar.google.com/citations?user=YY0qAeUAAAAJ&hl=zh-CN), [Zihao Chen](https://scholar.google.com/citations?user=tFtcUVAAAAAJ&hl=zh-CN), [ChaoFan Ding](), [Xinhan Di](https://scholar.google.com/citations?hl=en&user=CDijR8YAAAAJ&view_op=list_works&sortby=pubdate)


## Highlight

DeepSound-V1 is a framework enabling audio generation from videos towards initial step-by-step thinking without extra annotations based on the internal chain-of-thought (CoT) of Multi-modal large language model(MLLM).

## Main Results

Video-to-audio results on the VGGSound test set. The bold text highlights the superior performance of our proposed method
compared to previous methods, while the green text in brackets represents the improvement rate of each index.

|V2A Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Method&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| $FD_{PaSST}\downarrow$ | $FD_{PANNs}\downarrow$ | $FD_{VGG}\downarrow$ | $KL_{PANNs}\downarrow$ | $KL_{PaSST}\downarrow$ | $IS\uparrow$ | $IB\text{-}score\uparrow$ | $DeSync\downarrow$ |
|----------------------------------------|----------------------------------------|------------------------|------------------------|-----------------------|------------------------|------------------------|--------------|---------------------------|---------------------|
|**MMAudio-S-44k**| Direct \& Ori-Set | 65.25 | 5.55 | 1.66 | 1.67 | 1.44 | 18.02 | 32.27 | 0.444 |
|| Direct \& VO-Free  | 65.47 | 5.77  | 1.03 | 2.22  | 1.82  | 13.32  | 31.16 | 0.487|
|| Direct-neg \& Ori-Set | 68.44 | 6.48  | 1.71 | 2.27  | 1.84  | 13.74  | 30.51 | 0.505|
|| Our best \& VO-Free| **65.07(0.27%)** | 6.08  | **1.02(38.61%)**| 2.20  | 1.82  | 13.39  | 30.82 | 0.496|
|**MMAudio-M-44k**| Direct \& Ori-Set  | 61.88 | 4.74  | 1.13 | 1.66  | 1.41  | 17.41  | 32.99 | 0.443|
|| Direct \& VO-Free  | 56.07 | 4.57  | 0.99 | 2.15  | 1.74  | 13.91  | 32.19 | 0.479|
|| Direct-neg \& Ori-Set | 60.21 | 4.79  | 1.66 | 2.20  | 1.76  | 14.68  | 32.13 | 0.486|
|| Our best \& VO-Free| **55.65(10.07%)**| 4.80  | **0.93(17.70%)**| 2.15  | 1.77  | 13.82  | 31.44 | 0.495|
|**MMAudio-L-44k** | Direct \& Ori-Set  | 60.60 | 4.72  | 0.97 | 1.65  | 1.40  | 17.40  | 33.22 | 0.442|
|| Direct \& VO-Free  | 56.29 | 4.29  | 1.03 | 2.13  | 1.72  | 14.54  | 32.74 | 0.475|
|| Direct-neg \& Ori-Set | 59.50 | 4.62  | 1.75 | 2.19  | 1.76  | 15.42  | 32.36 | 0.490|
|| Our best \& VO-Free| **55.19(8.93%)** | **4.42(6.36%)**  | **0.95(2.06%)** | 2.13  | 1.75  | 14.49  | 31.94 | 0.490|
|**YingSound**| Direct \& Ori-Set  | 69.37 | 6.28  | 0.78 | 1.70  | 1.41  | 14.02  | 27.75 | 0.956|
|| Direct \& VO-Free  | 68.78 | 5.33  | 0.70 | 1.74  | 1.45  | 14.63  | 27.75 | 0.956|
|| Direct-neg \& Ori-Set | 77.86 | 7.37  | 0.75 | 2.20  | 1.83  | 12.48  | 27.15 | 0.991|
|| Our best \& VO-Free| **68.95(0.60%)** | **5.57(11.32%)** | **0.72(8.32%)** | 1.73  | 1.45  | **14.71(4.95%)** | 27.56 | 0.962|
|**FoleyCrafter**| Direct \& Ori-Set  | 140.09| 19.67 | 2.51 | 2.30  | 2.23  | 15.58  | 25.68 | 1.225|
|| Direct \& VO-Free  | 130.67| 17.59 | 2.12 | 2.59  | 2.28  | 9.94| 27.96 | 1.215|
|| Direct-neg \& Ori-Set | 181.45| 21.17 | 3.17 | 2.73  | 2.43  | 10.48  | 27.34 | 1.223|
|| Our best \& VO-Free| **127.97(8.65%)** | **17.39(11.62%)** | **2.12(15.42%)** | 2.57  | 2.29  | 9.96| **27.43(6.39%)** | **1.214(0.89%)** |





Ablation result on MMAudio-L-44k. The improvement between baseline and ours is represented as bold text, demonstrating effectiveness of the learned CoT reasoning in enhancing the final audio quality, the improvement between Ours-s3 and Ours-s4 is represented as bold and italic text.

| Method&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| $FD_{PaSST}\downarrow$ | $FD_{PANNs}\downarrow$ | $FD_{VGG}\downarrow$ | $KL_{PANNs}\downarrow$ | $KL_{PaSST}\downarrow$ | $IS\uparrow$ | $IB\text{-}score\uparrow$ | $DeSync\downarrow$ |
|----------------------------------------|----------------------------------------|------------------------|------------------------|-----------------------|------------------------|------------------------|--------------|---------------------------|
| **setting1 & vggtest-gt1** | 60.60 | 4.72 | 0.97 | 1.65 | 1.40 | 17.40 | 33.22 | 0.442 |
| **setting1 & vggtest-gt2** | 56.29 | 4.29 | 1.03 | 2.13 | 1.72 | 14.54 | 32.74 | 0.475 |
| **setting1-neg & vggtest-gt1** | 59.50 | 4.62 | 1.75 | 2.19 | 1.76 | 15.42 | 32.36 | 0.490 |
| **setting2-cot & vggtest-gt2** | **55.19(8.93%)** | **4.42(6.36%)** | **0.95(2.06%)** | 2.13 | 1.75 | 14.49 | 31.94 | 0.490 |
| **setting3-rm & vggtest-gt2** | 55.75 | 4.49 | 1.00 | 2.12 | 1.73 | 14.70 | 32.25 | 0.484 |
| **setting3-rep & vggtest-gt2** | **55.66(8.15%)** | **4.45(5.72%)** | **0.97(0.00%)** | 2.14 | ***1.74(0.57%)*** | ***14.61(0.83%)*** | ***32.16(0.69%)*** | ***0.486(0.82%)*** |
| **setting3-neg & vggtest-gt2** | 55.66 | 4.44 | 0.99 | 2.13 | 1.74 | 14.65 | 32.17 | 0.487 |




## Demo
Overall process:

https://github.com/user-attachments/assets/40767fd0-0d77-4286-9644-b5d66412652e


https://github.com/user-attachments/assets/a4e34ae2-4b8f-437d-9326-85851443805f

<!-- Direct result:

https://github.com/user-attachments/assets/a0e1ff4d-3bf8-47a3-9b35-a28c5654610d


Step-by-step result:

https://github.com/user-attachments/assets/96e65ec8-1631-4293-8c7f-244fc585e3aa -->


<table class='center'>
<tr>
  <td><p style="text-align: center">Direct Result</p></td>
  <td><p style="text-align: center">Step-by-step Result</p></td>
</tr>


<td>

https://github.com/user-attachments/assets/b69e6fd0-215e-4fdf-b60e-526b431fb4cc

</td>

<td style="width: 50%; padding: 10px;">

https://github.com/user-attachments/assets/96e65ec8-1631-4293-8c7f-244fc585e3aa

</td>

<tr>
<tr>

  <td>

https://github.com/user-attachments/assets/0662ad52-d46e-492a-b07c-d2e4df169167

</td>

  <td>

https://github.com/user-attachments/assets/a006e6b4-d108-4752-9d6a-33f8693d671a

</td>

<tr>
<tr>

  <td>

https://github.com/user-attachments/assets/21deb4a3-6464-4dc8-abf4-0c06f628480e

</td>

  <td>

https://github.com/user-attachments/assets/adbddacd-7161-4c4e-8211-d450c6724869

</td>

<tr>
<tr>

  <td>

https://github.com/user-attachments/assets/21deb4a3-6464-4dc8-abf4-0c06f628480e

</td>

  <td>

https://github.com/user-attachments/assets/b3b95546-82ea-4fb3-88cd-26c2e00f8356

</td>

</tr>
</table>




## Installation
```bash
conda create -n deepsound-v1 python=3.10 -y
conda activate deepsound-v1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 
pip install flash-attn==2.5.8 --no-build-isolation
pip install -r reqirments.txt
```

### Pretrained models
See [MODELS.md](docs/MODELS.md).

## Infer

### Quick Start
```bash
# coding = utf-8
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
mmaudio_path = os.path.join(project_root, 'third_party', 'MMAudio')
sys.path.append(mmaudio_path)

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
 step2_model_dir='pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT',
 step2_mode='cot',
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
def video_to_audio(pipeline: Pipeline, video_input, output_dir, mode='s4', postp_mode='neg', 
  prompt='', negative_prompt='', duration=10, skip_final_video=False):
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
  if skip_final_video:
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

Other parameters are [here](https://github.com/lym0302/DeepSound-V1/blob/main/demo.py#L25).




### Gradio interface

```bash
python gradio_demo.py
```



## Evaluation
Refer [av-benchmark](https://github.com/hkchengrex/av-benchmark) for benchmarking results.
See [EVAL.md](docs/EVAL.md).


## Citation

```bibtex
@misc{liang2025deepsoundv1startthinkstepbystep,
      title={DeepSound-V1: Start to Think Step-by-Step in the Audio Generation from Videos}, 
      author={Yunming Liang and Zihao Chen and Chaofan Ding and Xinhan Di},
      year={2025},
      eprint={2503.22208},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.22208}, 
}
```



## Acknowledgement

Many thanks to:
- [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) for MLLM CoT reasoning
- [MMAudio](https://github.com/hkchengrex/MMAudio) for generate audio from video
- [FoleyCrafter](https://github.com/open-mmlab/FoleyCrafter) for generate audio from video
- [BS-RoFormer](https://github.com/ZFTurbo/Music-Source-Separation-Training) for remove voice-over
- [av-benchmark](https://github.com/hkchengrex/av-benchmark) for benchmarking results
