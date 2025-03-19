# coding=utf-8
# CoT generate step-by-step

from third_party.VideoLLaMA2.videollama2 import model_init, mm_infer
import logging
    
class Step0:
    def __init__(self, model_path, modal_type='v'):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)

        self.model, self.processor, self.tokenizer = model_init(model_path)
        self.modal_type=modal_type
        if modal_type == "a":
            self.model.model.vision_tower = None
        elif modal_type == "v":
            self.model.model.audio_tower = None
        elif modal_type == "av":
            pass
        else:
            raise NotImplementedError
        self.modal = 'audio' if modal_type == "a" else "video"
        self.question = f"Generate high-quality audio from video step-by-step."
        self.preprocess = self.processor[self.modal]

    def run(self, video_path):
        self.log.info("Start to generate high-quality audio from video step-by-step...")
        audio_video_tensor = self.preprocess(video_path, va=False)
        output = mm_infer(
            audio_video_tensor,
            self.question,
            model=self.model,
            tokenizer=self.tokenizer,
            modal=self.modal,
            do_sample=False,
        )

        return output
