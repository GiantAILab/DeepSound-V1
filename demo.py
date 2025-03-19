# coding = utf-8
import argparse

from pipeline.pipeline import Pipeline
from models.mllm import MLLM
from models.v2a import V2A
from models.remove_vo import RemoveVO
from models.silence_det import SilenceDet
from third_party.MMAudio.mmaudio.eval_utils import setup_eval_logging
import os

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
    
    parser.add_argument('--gen_video', type=str, 
                        default='true', 
                        help="Whether to generate video, true means yes")
    
    parser.add_argument('--prompt', type=str, 
                        default='', 
                        help="prompt for v2a")
    
    parser.add_argument('--negative_prompt', type=str, 
                        default='', 
                        help="negative_prompt for v2a")
    
    parser.add_argument('--output_dir', type=str, 
                        default='outputs', 
                        help="output dir")
    
    parser.add_argument('--video_input', type=str, 
                        default='ZxiXftx2EMg_000477.mp4', 
                        # required=True,
                        help="video input path")
    
    return parser.parse_args()

# 示例使用
if __name__ == '__main__':
    args = parse_args()
    setup_eval_logging()

    pipeline = Pipeline(
        step0_model_dir=args.step0_model_dir, 
        step1_mode=args.step1_mode, 
        step2_model_dir=args.step2_model_dir,
        step2_mode=args.step2_mode,
        step3_mode=args.step3_mode,
    )

    # video_input = 'input_video.mp4'
    # output_dir = "outputs"
    os.makedirs(args.output_dir, exist_ok=True)
    step_results = pipeline.run(video_input=args.video_input, 
                                output_dir=args.output_dir,
                                mode=args.mode,
                                postp_mode=args.postp_mode,
                                prompt=args.prompt,
                                negative_prompt=args.negative_prompt)
    
    final_audio_path = step_results["final_audio_path"]
    final_video_path = step_results["final_video_path"]
    if args.gen_video:
        if step_results["final_audio_path"] is not None and step_results["final_video_path"] is None:
            # 这里混合视频加音频
            pass


    print(f"final_audio_path: {final_audio_path}")
    print(f"final_video_path: {final_video_path}")
