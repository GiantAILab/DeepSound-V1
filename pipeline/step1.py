# coding=utf-8
# V2A
import logging


class Step1:
    def __init__(self, step1_mode):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)

        if step1_mode.startswith('mmaudio'):
            from models.v2a_mmaudio import V2A_MMAudio
            variant = step1_mode.replace("mmaudio_", "")
            self.v2a_model = V2A_MMAudio(variant)


    def run(self, video_path, output_dir, prompt='', negative_prompt='', is_postp=False):
        self.log.info("Step1: Generate audio from video.")
        step1_audio_path, step1_video_path = self.v2a_model.generate_audio(
            video_path=video_path,
            output_dir=output_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            is_postp=is_postp)
        self.log.info("Finish Step1 successfuilly")
        return step1_audio_path, step1_video_path
