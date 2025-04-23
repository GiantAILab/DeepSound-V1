#coding=utf-8
import logging
import os
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
import os
from pathlib import Path

import soundfile as sf
import torch
import torchvision
from huggingface_hub import snapshot_download
from moviepy.editor import AudioFileClip, VideoFileClip
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from third_party.FoleyCrafter.foleycrafter.models.onset import torch_utils
from third_party.FoleyCrafter.foleycrafter.models.time_detector.model import VideoOnsetNet
from third_party.FoleyCrafter.foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from third_party.FoleyCrafter.foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy


vision_transform_list = [
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.CenterCrop((112, 112)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
video_transform = torchvision.transforms.Compose(vision_transform_list)

model_base_dir = "pretrained/v2a/foleycrafter"

class V2A_FoleyCrafter:
    def __init__(self, 
                pretrained_model_name_or_path: str=f"{model_base_dir}/checkpoints/auffusion",
                ckpt: str=f"{model_base_dir}/checkpoints",
                device=None):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        self.log.info(f"The V2A model uses FoleyCrafter, init...")
        
        if device is not None:
            self.device = device
        else:
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.log.warning('CUDA/MPS are not available, running on CPU')
        
        # download ckpt
        if not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path)


        # ckpt path
        temporal_ckpt_path = os.path.join(ckpt, "temporal_adapter.ckpt")

        # load vocoder
        self.vocoder = Generator.from_pretrained(ckpt, subfolder="vocoder").to(self.device)

        # load time_detector
        time_detector_ckpt = os.path.join(ckpt, "timestamp_detector.pth.tar")
        self.time_detector = VideoOnsetNet(False)
        self.time_detector, _ = torch_utils.load_model(time_detector_ckpt, self.time_detector, device=self.device, strict=True)

        # load adapters
        self.pipe = build_foleycrafter().to(self.device)
        ckpt = torch.load(temporal_ckpt_path)

        # load temporal adapter
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        load_gligen_ckpt = {}
        for key, value in ckpt.items():
            if key.startswith("module."):
                load_gligen_ckpt[key[len("module.") :]] = value
            else:
                load_gligen_ckpt[key] = value
        m, u = self.pipe.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
        print(f"### Control Net missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        # load semantic adapter
        self.pipe.load_ip_adapter(
            os.path.join(ckpt, "semantic"), subfolder="", weight_name="semantic_adapter.bin", image_encoder_folder=None
        )
        # ip_adapter_weight = semantic_scale
        # self.pipe.set_ip_adapter_scale(ip_adapter_weight)

        self.generator = torch.Generator(device=self.device)
        # self.generator.manual_seed(seed)
        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", subfolder="models/image_encoder"
        ).to(self.device)

    @torch.no_grad()
    def generate_audio(self, 
                       video_path,
                       output_dir,
                       prompt: str='', 
                       negative_prompt: str='',
                       seed: int=42,
                       temporal_scale: float=0.2,
                       semantic_scale: float=1.0,
                       is_postp=False,):
        
        self.pipe.set_ip_adapter_scale(semantic_scale)
        self.generator.manual_seed(seed)
        
        video_path = Path(video_path).expanduser()
        output_dir = Path(output_dir).expanduser()
        self.log.info(f"Loading video: {video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        frames, duration = read_frames_with_moviepy(video_path, max_frame_nums=150)
        time_frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
        time_frames = video_transform(time_frames)
        time_frames = {"frames": time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)}
        preds = self.time_detector(time_frames)
        preds = torch.sigmoid(preds)
        time_condition = [
            -1 if preds[0][int(i / (1024 / 10 * duration) * 150)] < 0.5 else 1
            for i in range(int(1024 / 10 * duration))
        ]
        time_condition = time_condition + [-1] * (1024 - len(time_condition))
        # w -> b c h w
        time_condition = (
            torch.FloatTensor(time_condition)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 1, 256, 1)
            .to("cuda")
        )
        images = self.image_processor(images=frames, return_tensors="pt").to("cuda")
        image_embeddings = self.image_encoder(**images).image_embeds
        image_embeddings = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
        neg_image_embeddings = torch.zeros_like(image_embeddings)
        image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1)


        sample = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=image_embeddings,
            image=time_condition,
            controlnet_conditioning_scale=temporal_scale,
            num_inference_steps=25,
            height=256,
            width=1024,
            output_type="pt",
            generator=self.generator,
        )

        audio_img = sample.images[0]
        audio = denormalize_spectrogram(audio_img)
        audio = self.vocoder.inference(audio, lengths=160000)[0]
        audio = audio[: int(duration * 16000)]
        
        if is_postp:
            audio_save_path = output_dir / f'{video_path.stem}.neg.wav'
            video_save_path = output_dir / f'{video_path.stem}.neg.mp4'
        else:
            audio_save_path = output_dir / f'{video_path.stem}.step1.wav'
            video_save_path = output_dir / f'{video_path.stem}.step1.mp4'

        
        self.log.info(f"Saving generated audio and video to {output_dir}")
        sf.write(audio_save_path, audio, 16000)

        audio = AudioFileClip(audio_save_path)
        video = VideoFileClip(video_path)
        duration = min(audio.duration, video.duration)
        audio = audio.subclip(0, duration)
        video.audio = audio
        video = video.subclip(0, duration)
        video.write_videofile(video_save_path)
        self.log.info(f'Video saved to {video_save_path}')

        return audio_save_path, video_save_path
