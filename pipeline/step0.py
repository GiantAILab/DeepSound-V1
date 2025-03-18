# coding=utf-8
# CoT generate step-by-step

from third_party.VideoLLaMA2.videollama2 import model_init, mm_infer
    
class Step0:
    def __init__(self, model_path):
        self.model, self.processor, self.tokenizer = model_init(model_path)
        self.model.model.audio_tower = None
        self.modal_type="v"
        self.modal = "video"
        self.question = f"Generate high-quality audio from video step-by-step."
        self.preprocess = self.processor[self.modal]

    def run(self, video_path):
        print("[Step0] Generate high-quality audio from video step-by-step...")
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
