# coding=utf-8

from step0 import Step0
from step1 import Step1
from step2 import Step2
from step3 import Step3
from step4 import Step4

class Pipeline:
    def __init__(self, step0_model_dir, mllm_step2_processor, v2a_processor,
                 remove_vo_processor, silence_detector, mode='s4', step2_mode='cot', postp_mode='rep'):
        self.step0 = Step0(step0_model_dir)
        self.step1 = Step1(v2a_processor)
        self.step2 = Step2(mllm_step2_processor, step2_mode)
        self.step3 = Step3(remove_vo_processor)
        self.step4 = Step4(silence_detector)

    def run(self, video_input):
        print("[Pipeline] ====== 开始处理 ======")

        step1_info, step2_info, step3_info, step4_info, mode = self.step0.run(video_input)
        generated_audio, video_with_audio = self.step1.run(step1_info)
        has_voiceover = self.step2.run(video_with_audio)
        clean_audio = self.step3.run(generated_audio, has_voiceover)

        if mode == 's3':
            print("[Pipeline] s3 模式，返回 step3 结果")
            return clean_audio

        final_audio = self.step4.run(clean_audio)

        print("[Pipeline] ====== 处理结束 ======")
        return final_audio
