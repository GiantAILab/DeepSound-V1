import os
import sys
import time
import torch
import gradio as gr
import subprocess
from pathlib import Path

from moviepy.editor import AudioFileClip, VideoFileClip

# 项目路径配置
project_root = os.path.dirname(os.path.abspath(__file__))
mmaudio_path = os.path.join(project_root, 'third_party', 'MMAudio')
sys.path.append(mmaudio_path)

from pipeline.pipeline import Pipeline
from third_party.MMAudio.mmaudio.eval_utils import setup_eval_logging

# 初始化模型（只加载一次）
setup_eval_logging()
pipeline = Pipeline(
    step0_model_dir='pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT', 
    step1_mode='mmaudio_large_44k', 
    step2_model_dir='pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT',
    step2_mode='cot',
    step3_mode='bs_roformer',
)
output_dir = "output_gradio"
os.makedirs(output_dir, exist_ok=True)
skip_final_video = False
def video_to_audio(
        video_input: gr.Video,
        prompt: str='', 
        negative_prompt: str='',
        mode: str='s4',
        postp_mode: str='neg',
        duration: float=10,
        seed: int=42,):
    """
    视频转音频的推理流程。
    """

    st_infer = time.time()
    video_input = str(video_input)

    # pipeline 推理
    step_results = pipeline.run(
        video_input=video_input, 
        output_dir=output_dir,
        mode=mode,
        postp_mode=postp_mode,
        prompt=prompt,
        negative_prompt=negative_prompt,
        duration=duration,
        seed=seed
    )
    
    temp_final_audio_path = step_results["temp_final_audio_path"]
    temp_final_video_path = step_results["temp_final_video_path"]

    video_name_stem = Path(video_input).stem
    final_audio_path = str(Path(output_dir) / f'{video_name_stem}.wav')
    final_video_path = str(Path(output_dir) / f'{video_name_stem}.mp4')

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
    print("step_results: ", step_results)

    # return final_audio_path, final_video_path
    return final_video_path


video_to_audio_tab = gr.Interface(
    fn=video_to_audio,
    # Project page: <a href="https://hkchengrex.com/MMAudio/">https://hkchengrex.com/MMAudio/</a><br>
    description="""
    Code: <a href="https://github.com/lym0302/DeepSound-V1">https://github.com/lym0302/DeepSound-V1</a><br>

    NOTE: It takes longer to process high-resolution videos (>384 px on the shorter side). 
    Doing so does not improve results.

    This is a step-by-step v2a process and may take a long time. 
    If Post Processing is set to 'rm', the generated video may be None.
    """,
    inputs=[
        gr.Video(),
        gr.Text(label='Prompt'),
        gr.Text(label='Negative prompt', value=''),
        gr.Radio(["s3", "s4"], label="Mode", value="s4"),
        gr.Radio(["rm", "rep", "neg"], label="Post Processing", value="neg"),
        gr.Number(label='Duration (sec)', value=10, minimum=1),
        gr.Number(label='Seed (42: random)', value=42, precision=0, minimum=-1),
        # gr.Checkbox(label="Skip Final Video", value=False),
    ],
    # outputs=[gr.Audio(label="Generated Audio"), gr.Video(label="Generated Video")],  # 输出音频文件和视频文件
    outputs=[gr.Video(label="Generated Video")],
    cache_examples=False,
    title='DeepSound-V1 — Video-to-Audio Synthesis',
)


# if __name__ == "__main__":
#     gr.TabbedInterface([video_to_audio_tab],
#                        ['Video-to-Audio']).launch(allowed_paths=[output_dir])


if __name__ == "__main__":
    port = 8000

    gr.TabbedInterface([video_to_audio_tab, ],
                       ['Video-to-Audio', ]).launch(
                           server_port=port, allowed_paths=[output_dir])
