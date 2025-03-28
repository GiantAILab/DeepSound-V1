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

    pipeline = Pipeline(
        step0_model_dir=step0_model_dir, 
        step1_mode=step1_mode, 
        step2_model_dir=step2_model_dir,
        step2_mode=step2_mode,
        step3_mode=step3_mode,
    )

    return pipeline


@torch.inference_mode()
def video_to_audio(pipeline: Pipeline, video_input, output_dir, mode='s4', postp_mode='neg', 
                   prompt='', negative_prompt='', duration=10, skip_final_video=False):
    st_infer = time.time()
    step_results = pipeline.run_for_show(video_input=video_input, 
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
    # print(f"Inference time: {et_infer - st_infer:.2f} s.")
    return step_results


@torch.inference_mode()
def main():
    setup_eval_logging()
    video_input = "aa.mp4"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = init_pipeline()
    print("\nModel initialization is completed.")
    step_results = video_to_audio(pipeline, video_input, output_dir)
    print("\nGenerating audio from video is complete.")

    # print("step_results: ", step_results)
    # print(f"final_audio_path: {step_results['final_audio_path']}")
    # print(f"final_video_path: {step_results['final_video_path']}")


if __name__ == '__main__':
    main()