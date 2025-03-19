# coding=utf-8

from .step0 import Step0
from .step1 import Step1
from .step2 import Step2
from .step3 import Step3
from .step4 import Step4
import logging
import re
import os

class Pipeline:
    def __init__(self, step0_model_dir, step1_mode, step2_model_dir, step2_mode, step3_mode):
        self.step0 = Step0(step0_model_dir)
        self.step1 = Step1(step1_mode)
        self.step2 = Step2(step2_model_dir, step2_mode)
        self.step3 = Step3(model_type=step3_mode)
        self.step4 = Step4()
        self.step_processors = [self.step1, self.step2, self.step3, self.step4]
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        

    def run(self, video_input, output_dir, mode='s4', postp_mode='rep', prompt='', negative_prompt='', duration=10):
        step0_resp = self.step0.run(video_input)
        step0_resp_list = re.findall(r'(Step\d:.*?)(?=Step\d:|$)', step0_resp, re.DOTALL)
        step_infos = [step_info.strip().split("\n")[0] for step_info in step0_resp_list]
        step3_temp_dir = os.path.join(output_dir, "remove_vo")
        
        step_results = {}
        step_results["final_audio_path"] = None
        step_results["final_video_path"] = None
        for step_info in step_infos:
            self.log.info(f"Start to {step_info}")
            if step_info == 'Step1: Generate audio from video.':
                step1_audio_path, step1_video_path = self.step1.run(video_input, output_dir, prompt, negative_prompt, duration=duration)
                step_results["step1_audio_path"] = step1_audio_path
                step_results["step1_video_path"] = step1_video_path
            elif step_info == 'Step2: Given a video and its generated audio, determine whether the audio contains voice-over.':
                is_vo = self.step2.run(step_results["step1_video_path"])
                step_results["is_vo"] = is_vo
            elif step_info == 'Step3: Remove voice-over from audio.':
                if step_results["is_vo"] == "A":  # is voice-over
                    step3_audio_path = self.step3.run(input_audio_path=step_results["step1_audio_path"],
                                   temp_store_dir=step3_temp_dir,
                                   output_dir=output_dir)
                    step_results["step3_audio_path"] = step3_audio_path
                    if mode == 's3':
                        step_results["final_audio_path"] = step_results["step3_audio_path"]
                        return step_results
                else:
                    step_results["final_audio_path"] = step_results["step1_audio_path"]
                    step_results["final_video_path"] = step_results["step1_video_path"]
                    return step_results
            elif step_info == 'Step4: Determine whether the audio is silent.':
                is_silent = self.step4.run(step_results["step3_audio_path"])
                step_results["is_silent"] = is_silent

        # post-process
        if not step_results["is_silent"]:  # if not silent
            step_results["final_audio_path"] = step_results["step3_audio_path"]
            return step_results
        else:
            if postp_mode == "rm":
                step_results["final_audio_path"] = None
                return step_results
            elif postp_mode == "rep":
                step_results["final_audio_path"] = step_results["step1_audio_path"]
                step_results["final_video_path"] = step_results["step1_video_path"]
                return step_results
            elif postp_mode == "neg":
                neg_audio_path, neg_video_path = self.step1.run(video_input, output_dir, prompt, negative_prompt='huamn voice', duration=duration, is_postp=True)
                step_results["final_audio_path"] = neg_audio_path
                step_results["final_video_path"] = neg_video_path
                return step_results
            
        return step_results

