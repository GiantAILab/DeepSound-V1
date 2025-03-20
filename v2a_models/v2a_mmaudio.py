#coding=utf-8
import logging
from pathlib import Path
import torch
import torchaudio


from third_party.MMAudio.mmaudio.eval_utils import ModelConfig, all_model_cfg, generate, load_video, make_video, setup_eval_logging
from third_party.MMAudio.mmaudio.model.flow_matching import FlowMatching
from third_party.MMAudio.mmaudio.model.networks import MMAudio, get_my_mmaudio
from third_party.MMAudio.mmaudio.model.utils.features_utils import FeaturesUtils


class V2A_MMAudio:
    def __init__(self, 
                variant: str="large_44k",
                num_steps: int=25,
                seed: int=42,
                full_precision: bool=False,):
        
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info(f"The V2A model uses MMAudio {variant}, init...")
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.log.warning('CUDA/MPS are not available, running on CPU')
        self.dtype = torch.float32 if full_precision else torch.bfloat16

        if variant not in all_model_cfg:
            raise ValueError(f'Unknown model variant: {variant}')
        self.model: ModelConfig = all_model_cfg[variant]
        self.model.download_if_needed()

        self.net: MMAudio= get_my_mmaudio(self.model.model_name).to(self.device, self.dtype).eval()
        self.net.load_weights(torch.load(self.model.model_path, map_location=self.device, weights_only=True))

        # Setup random generator for reproducibility
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)

        # Flow Matching
        self.fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

        # Feature utils setup
        self.feature_utils = FeaturesUtils(tod_vae_ckpt=self.model.vae_path,
                                           synchformer_ckpt=self.model.synchformer_ckpt,
                                           enable_conditions=True,
                                           mode=self.model.mode,
                                           bigvgan_vocoder_ckpt=self.model.bigvgan_16k_path,
                                           need_vae_encoder=False)
        self.feature_utils = self.feature_utils.to(self.device, self.dtype).eval()


    def generate_audio(self, 
                       video_path,
                       output_dir,
                       prompt: str='', 
                       negative_prompt: str='',
                       duration: int=10,
                       cfg_strength: float=4.5,
                       mask_away_clip: bool=False,
                       is_postp=False,):
        
        video_path = Path(video_path).expanduser()
        output_dir = Path(output_dir).expanduser()
        self.log.info(f"Loading video: {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec

        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)

        seq_cfg = self.model.seq_cfg
        seq_cfg.duration = duration
        self.net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        self.log.info(f'Prompt: {prompt}')
        self.log.info(f'Negative prompt: {negative_prompt}')
        
        self.log.info(f"Generating Audio...")
        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            negative_text=[negative_prompt],
            feature_utils=self.feature_utils,
            net=self.net,
            fm=self.fm,
            rng=self.rng,
            cfg_strength=cfg_strength)
        audio = audios.float().cpu()[0]
        
        if is_postp:
            audio_save_path = output_dir / f'{video_path.stem}.neg.wav'
            video_save_path = output_dir / f'{video_path.stem}.neg.mp4'
        else:
            audio_save_path = output_dir / f'{video_path.stem}.step1.wav'
            video_save_path = output_dir / f'{video_path.stem}.step1.mp4'

        
        self.log.info(f"Saving generated audio and video to {output_dir}")
        torchaudio.save(str(audio_save_path), audio, seq_cfg.sampling_rate)
        self.log.info(f'Audio saved to {audio_save_path}')
        make_video(video_info, str(video_save_path), audio, sampling_rate=seq_cfg.sampling_rate)
        self.log.info(f'Video saved to {video_save_path}')

        return audio_save_path, video_save_path



# def main():
#     # 初始化日志（如果你有 logger.py，推荐只做一次初始化）
#     setup_eval_logging()

#     # 初始化模型
#     v2a_model = V2A_MMAudio(
#         variant="large_44k_v2",     # 这个是你模型的版本名
#         num_steps=25,               # 采样步数
#         seed=42,                    # 随机种子
#         full_precision=False        # 是否使用全精度
#     )

#     # 视频路径（换成你的真实路径）
#     video_path = "ZxiXftx2EMg_000477.mp4"

#     # 输出目录
#     output_dir = "outputs"

#     # 提示词（控制生成内容）
#     prompt = ""
#     negative_prompt = ""

#     # 生成音频 + 视频
#     audio_save_path, video_save_path = v2a_model.generate_audio(
#         video_path=video_path,
#         output_dir=output_dir,
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         duration=10,            # 秒
#         cfg_strength=4.5,       # 指导强度
#         mask_away_clip=False    # 是否移除 clip
#     )

# if __name__ == "__main__":
#     main()