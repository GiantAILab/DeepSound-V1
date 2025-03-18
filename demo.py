# coding = utf-8
import argparse

from pipeline.pipeline import Pipeline
from models.mllm import MLLM
from models.v2a import V2A
from models.remove_vo import RemoveVO
from models.silence_det import SilenceDet
from third_party.MMAudio.mmaudio.eval_utils import setup_eval_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the model configuration")
    parser.add_argument('--mode', type=str, 
                        choices=['s3', 's4'], 
                        default='s4', 
                        help="until step3 or step4")
    parser.add_argument('--step1_mode', type=str, 
                        choices=['mmaudio_small_16k', 'mmaudio_small_44k', 'mmaudio_medium_44k', 'mmaudio_large_44k', 'mmaudio_large_44k_v2' 'foleycrafter'], 
                        default='mmaudio_large_44k_v2', 
                        help='v2a method')
    parser.add_argument('--step2_mode', type=str, choices=['qa', 'cot'], 
                        default='cot', 
                        help="judge voice-over method")
    parser.add_argument('--step3_mode', type=str, 
                        choices=['bs_roformer'], 
                        default='bs_roformer', 
                        help="remove voice-over method")
    parser.add_argument('--postp_mode', type=str, 
                        choices=['rm', 'rep', 'neg'], 
                        default='rep', 
                        help="post-process method")
    parser.add_argument('--step0_model_dir', type=str, 
                        default='pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT', 
                        help="step0 model dir")
    parser.add_argument('--step1_model_dir', type=str, 
                        default='pretrained/v2a/mmaudio', 
                        help="v2a model dir")
    parser.add_argument('--step2_model_dir', type=str, 
                        default='pretrained/mllm/VideoLLaMA2.1-7B-AV-CoT', 
                        help="judge voice-over model dir")
    parser.add_argument('--step3_model_dir', type=str, 
                        default='pretrained/remove_vo/checkpoints', 
                        help="remove voice-over model path")
    
    return parser.parse_args()

# 示例使用
if __name__ == '__main__':
    args = parse_args()
    setup_eval_logging()
    # mllm_step0_processor, mllm_step2_processor = MLLM(args.step0_model_dir, args.step2_model_dir)
    v2a_processor = V2A(args.step1_mode, args.step1_model_dir)
    remove_vo_processor = RemoveVO(args.step3_mode, args.step3_model_dir)
    silence_detector = SilenceDet()

    pipeline = Pipeline(
        args.step0_model_dir,
        mllm_step2_processor,
        v2a_processor,
        remove_vo_processor,
        silence_detector,
        mode=args.mode,
        step2_mode=args.step2_mode,
        postp_mode=args.postp_mode,
    )

    video_input = 'input_video.mp4'
    final_result = pipeline.run(video_input)

    print(f"最终生成结果: {final_result}")
