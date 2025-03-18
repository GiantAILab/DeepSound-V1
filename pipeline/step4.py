# coding=utf-8
# Silence detection 

class Step4:
    def __init__(self, silence_detector, handle_type='rm'):
        self.silence_detector = silence_detector
        self.handle_type = handle_type  # 'rm', 'rep', or 'neg'

    def run(self, clean_audio):
        print("[Step4] 判断是否为静音...")
        is_silence = self.silence_detector.check_silence(clean_audio)
        print(f"[Step4] 是否静音: {is_silence}")

        if is_silence:
            print(f"[Step4] 静音处理方式: {self.handle_type}")
            if self.handle_type == 'rm':
                print("[Step4] 处理为 rm，返回 None")
                return None
            elif self.handle_type == 'rep':
                print("[Step4] 处理为 rep，返回替换音频")
                return self.silence_detector.replace_audio(clean_audio)
            elif self.handle_type == 'neg':
                print("[Step4] 处理为 neg，返回负样本音频")
                return self.silence_detector.negative_audio(clean_audio)
            else:
                raise ValueError(f"未知处理类型: {self.handle_type}")
        else:
            print("[Step4] 音频正常，直接返回")
            return clean_audio
