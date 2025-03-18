# coding=utf-8
# Remove voice-over

class Step3:
    def __init__(self, bsroformer_model, mode='s4'):
        self.bsroformer_model = bsroformer_model
        self.mode = mode  # 's3' or 's4'

    def run(self, generated_audio, has_voiceover):
        print("[Step3] 进入画外音处理流程...")
        if has_voiceover:
            print("[Step3] 检测到画外音，进行去除...")
            clean_audio = self.bsroformer_model.remove_voiceover(generated_audio)
            print("[Step3] 去除画外音完成")
        else:
            print("[Step3] 无画外音，直接使用生成音频")
            clean_audio = generated_audio

        if self.mode == 's3':
            print("[Step3] 模式为 s3，返回当前 clean_audio 作为最终结果")
            return clean_audio

        print("[Step3] 模式为 s4，继续下一步处理")
        return clean_audio
