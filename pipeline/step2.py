# coding=utf-8
# judge voice-over

from third_party.VideoLLaMA2.videollama2 import model_init, mm_infer
import logging
    
class Step2:
    def __init__(self, model_path, step2_mode, modal_type="av"):
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
        
        self.question = f"Given a video and its corresponding audio, determine whether the audio contains voice-over? Options: A. Yes, B. No. Choose A or B."
        self.preprocess = self.processor[self.modal]

        self.step2_mode = step2_mode

    def run(self, video_audio_path):
        self.log.info("Step2: Given a video and its generated audio, determine whether the audio contains voice-over.")
        audio_video_tensor = self.preprocess(video_audio_path, va=True)
        output = mm_infer(
            audio_video_tensor,
            self.question,
            model=self.model,
            tokenizer=self.tokenizer,
            modal=self.modal,
            do_sample=False,
        )
        self.log.info("Finish Step2 successfuilly")
        if self.step2_mode == "cot":
            output = output.split("<CONCLUSION>")[-1][1]
        return output
