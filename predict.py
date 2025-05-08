# Prediction interface for Cog ⚙️ 
# https://cog.run/python

from cog import BasePredictor, Input
from cog import Path as CogPath
from typing import Dict, Optional
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
mmaudio_path = os.path.join(project_root, 'third_party', 'MMAudio')
sys.path.append(mmaudio_path)

from pipeline.pipeline import Pipeline
import os
from moviepy.editor import AudioFileClip, VideoFileClip
import torch
from pathlib import Path
import subprocess
import requests

from pipeline.pipeline import Pipeline
import shutil
# if torch.cuda.is_available():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     load_8bit = True
# else:
#     load_8bit = False


class Predictor(BasePredictor):
    def setup(self) -> None:
        #repo_local_path = self.download_model()
        repo_local_path = 'pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT'
        self.step0_model_dir=repo_local_path
        self.step1_mode='mmaudio_medium_44k'
        self.step2_model_dir=repo_local_path
        self.step2_mode='cot'
        self.step3_mode='bs_roformer'
        
        self.pipeline = self.init_pipeline()


    def download_model(self):
        from huggingface_hub import snapshot_download
        repo_local_path = snapshot_download(repo_id="lym0302/VideoLLaMA2.1-7B-AV-CoT")

        remove_vo_model_dir = "pretrained/remove_vo/checkpoints"
        os.makedirs(remove_vo_model_dir, exist_ok=True)
        urls = ["https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"]
        for url in urls:
            file_name = url.split("/")[-1]  # Extract file name from URL
            file_path = os.path.join(remove_vo_model_dir, file_name)
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):  # Use a chunk size of 8 KB
                        f.write(chunk)
                print(f"File downloaded successfully and saved to {file_path}")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")

        os.makedirs("pretrained/v2a/mmaudio", exist_ok=True)
        return repo_local_path
    
    @torch.inference_mode()
    def init_pipeline(self):
        pipeline = Pipeline(
            step0_model_dir=self.step0_model_dir, 
            step1_mode=self.step1_mode, 
            step2_model_dir=self.step2_model_dir,
            step2_mode=self.step2_mode,
            step3_mode=self.step3_mode,
        )

        return pipeline
    
    @torch.inference_mode()
    def predict(self, 
                video_input: Optional[CogPath] = Input(description="Input video for processing"), 
                #output_dir: Path = Input(description="Directory to save the output video"),
                #mode: str = Input(description="Processing mode (e.g., 's4')", default="s4", choices=['s3', 's4']),
                #postp_mode: str = Input(description="Post-processing mode (e.g., 'neg')", default="neg", choices=['rep', 'neg', '']),
                prompt: str = Input(description="Prompt for video generation", default=""),
                negative_prompt: str = Input(description="Negative prompt for video generation", default=""),
                duration: float = Input(description="Duration of output video in seconds", default=10),
                skip_final_video: bool = Input(description="Flag to skip the final video processing", default=False)
                ) -> CogPath:
                #) -> Dict[str, CogPath]:
        video_input = str(video_input)
        mode = 's4'
        postp_mode = 'neg'
        output_dir = 'output'      
          
        step_results = self.pipeline.run(video_input=video_input, 
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
        
        print("step_results: ", step_results)
        #return CogPath(final_video_path)
        #shutil.copy(step_results["step1_video_path"], "output/play_step1_video.mp4")
        #shutil.copy(step_results["final_video_path"], "output/play_final_video.mp4")

        html_path = "output/result.html"
        with open(html_path, "w") as f:
            f.write(f"""
            <html>
            <body>
                <h2>Step 1 Video</h2>
                <video controls width="640">
                    <source src="output/play_step1_video.mp4" type="video/mp4">
                </video>
                <h2>Final Video</h2>
                <video controls width="640">
                    <source src="output/play_final_video.mp4" type="video/mp4">
                </video>
            </body>
            </html>
            """)
        #return {
        #    'step1_video': CogPath(step_results["step1_video_path"]),
        #    'final_video': CogPath(step_results["final_video_path"])
        #}
        #return CogPath(html_path)
        #return step_results["step1_video_path"], step_results["final_video_path"]
        return CogPath(final_video_path)
    
    

# pred = Predictor()
# video_input = "aa.mp4"
# output_dir = "output"
# os.makedirs(output_dir, exist_ok=True)
# step_results = pred.predict(video_input, output_dir)
# print(step_results)
