# coding=utf-8
# Remove voice-over
import logging
import argparse
import subprocess
import librosa
import os
import torch
import soundfile as sf
import numpy as np


# Using the embedded version of Python can also correctly import the utils module.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

from third_party.MusicSourceSeparationTraining.utils import demix, load_config, normalize_audio, denormalize_audio, draw_spectrogram
from third_party.MusicSourceSeparationTraining.utils import prefer_target_instrument, apply_tta, load_start_checkpoint
from third_party.MusicSourceSeparationTraining.models.bs_roformer import BSRoformer
import warnings

warnings.filterwarnings("ignore")

model_base_dir = "pretrained/remove_vo/checkpoints"
MODEL_PATHS = {"bs_roformer": [f"{model_base_dir}/model_bs_roformer_ep_317_sdr_12.9755.ckpt", f"{model_base_dir}/model_bs_roformer_ep_317_sdr_12.9755.yaml"]}


class Step3:
    def __init__(self, model_type="bs_roformer", device=None):
        model_path, config_path = MODEL_PATHS[model_type]
        
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        if device is not None:
            self.device=device
        else:
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.log.warning('CUDA/MPS are not available, running on CPU')
        
        self.model_type = model_type

        # self.model, self.config = get_model_from_config(model_type, config_path)
        self.config = load_config(model_type, config_path)
        self.model = BSRoformer(**dict(self.config.model))
        args = argparse.Namespace()
        args.start_check_point = model_path
        args.model_type = model_type
        args.lora_checkpoint = ''
        load_start_checkpoint(args, self.model, type_='inference')
        self.model = self.model.to(self.device)
        self.sample_rate = getattr(self.config.audio, 'sample_rate', 44100)

        
    def run(self,
            input_audio_path,
            temp_store_dir,  # for remove result dir
            output_dir,  # for final dir
            disable_detailed_pbar: bool=False,
            use_tta: bool= False,
            extract_instrumental: bool=True,
            codec="wav",
            subtype="FLOAT",
            draw_spectro=0,
            ):
        
        # self.log.info("Step3: Remove voice-over from audio.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if disable_detailed_pbar:
            detailed_pbar = False
        else:
            detailed_pbar = True

        instruments = prefer_target_instrument(self.config)[:]
        
        mix, sr = librosa.load(input_audio_path, sr=self.sample_rate, mono=False)
        # If mono audio we must adjust it depending on model
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in self.config.audio:
                if self.config.audio['num_channels'] == 2:
                    print(f'Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in self.config.inference:
            if self.config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        waveforms_orig = demix(self.config, self.model, mix, self.device, model_type=self.model_type, pbar=detailed_pbar)
        if use_tta:
            waveforms_orig = apply_tta(self.config, self.model, mix, waveforms_orig, self.device, self.model_type)

        if extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        file_name = os.path.splitext(os.path.basename(input_audio_path))[0].replace(".step1", "")
        temp_output_dir = os.path.join(temp_store_dir, file_name)
        os.makedirs(temp_output_dir, exist_ok=True)

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in self.config.inference:
                if self.config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            output_path = os.path.join(temp_output_dir, f"{instr}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            if draw_spectro > 0:
                output_img_path = os.path.join(temp_output_dir, f"{instr}.jpg")
                draw_spectrogram(estimates.T, sr, draw_spectro, output_img_path)


        instrumental_file = os.path.join(temp_output_dir, 'instrumental.wav')
        step3_audio_path = f"{output_dir}/{file_name}.step3.wav"
        subprocess.run(['cp', instrumental_file, step3_audio_path])

        self.log.info(f"The voice-over has been removed, and the audio is saved in {step3_audio_path}")
        self.log.info("Finish Step3 successfully.\n")
        return step3_audio_path



