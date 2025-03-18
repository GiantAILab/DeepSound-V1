# coding=utf-8
# judge voice-over

class Step2:
    def __init__(self, vllm_model, model_type='cot'):
        self.vllm_model = vllm_model
        self.model_type = model_type  # 'cot' or 'qa'

    def run(self, video_with_audio):
        print(f"[Step2] 使用 {self.model_type} 判断是否有画外音...")
        has_voiceover = self.vllm_model.judge_voiceover(video_with_audio, mode=self.model_type)
        print(f"[Step2] 判断完成，是否包含画外音: {has_voiceover}")
        return has_voiceover
