# Pretrained models

## Multi-Language-Model
### VideoLLaMA2
| Model    | Download link | File size |
| -------- | ------- | ------- |
| MLLM, step-by-step, CoT | <a href="https://huggingface.co/lym0302/VideoLLaMA2.1-7B-AV-CoT">MLLM-CoT</a> | ~17G |
| MLLM, QA | <a href="https://huggingface.co/lym0302/VideoLLaMA2.1-7B-AV-QA">MLLM-QA</a> | ~17G |

Download to `pretrained/mllm/`.


## Video-to-Audio

### MMAudio
<!-- The models will be downloaded automatically when you run the demo script. MD5 checksums are provided in `mmaudio/utils/download_utils.py`. -->
The models are also available at https://huggingface.co/hkchengrex/MMAudio/tree/main

| Model    | Download link | File size |
| -------- | ------- | ------- |
| Flow prediction network, small 16kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_16k.pth" download="mmaudio_small_16k.pth">mmaudio_small_16k.pth</a> | 601M |
| Flow prediction network, small 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_44k.pth" download="mmaudio_small_44k.pth">mmaudio_small_44k.pth</a> | 601M |
| Flow prediction network, medium 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_medium_44k.pth" download="mmaudio_medium_44k.pth">mmaudio_medium_44k.pth</a> | 2.4G |
| Flow prediction network, large 44.1kHz | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k.pth" download="mmaudio_large_44k.pth">mmaudio_large_44k.pth</a> | 3.9G |
| Flow prediction network, large 44.1kHz, v2 **(recommended)** | <a href="https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth" download="mmaudio_large_44k_v2.pth">mmaudio_large_44k_v2.pth</a> | 3.9G |
| 16kHz VAE | <a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth">v1-16.pth</a> | 655M |
| 16kHz BigVGAN vocoder (from Make-An-Audio 2) |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt">best_netG.pt</a> | 429M |
| 44.1kHz VAE |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth">v1-44.pth</a> | 1.2G | 
| Synchformer visual encoder |<a href="https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth">synchformer_state_dict.pth</a> | 907M |

<!-- To run the model, you need four components: a flow prediction network, visual feature extractors (Synchformer and CLIP, CLIP will be downloaded automatically), a VAE, and a vocoder. VAEs and vocoders are specific to the sampling rate (16kHz or 44.1kHz) and not model sizes.
The 44.1kHz vocoder will be downloaded automatically.
The `_v2` model performs worse in benchmarking (e.g., in  Fréchet distance), but, in my experience, generalizes better to new data. -->

The expected directory structure (full):

```bash
pretrained/v2a/mmaudio
├── ext_weights
│   ├── best_netG.pt
│   ├── synchformer_state_dict.pth
│   ├── v1-16.pth
│   └── v1-44.pth
├── weights
│   ├── mmaudio_small_16k.pth
│   ├── mmaudio_small_44k.pth
│   ├── mmaudio_medium_44k.pth
│   ├── mmaudio_large_44k.pth
│   └── mmaudio_large_44k_v2.pth
└── ...
```

The expected directory structure (minimal, for the `mmaudio_large_44k` model only):

```bash
pretrained/v2a/mmaudio
├── ext_weights
│   ├── synchformer_state_dict.pth
│   └── v1-44.pth
├── weights
│   └── mmaudio_large_44k.pth
└── ...
```


### FoleyCrafter
| Model    | Download link | File size |
| -------- | ------- | ------- |
| auffusion | <a href="https://huggingface.co/auffusion/auffusion-full-no-adapter">auffusion</a> | ~5G |
| FoleyCrafter | <a href="https://huggingface.co/ymzhang319/FoleyCrafter">FoleyCrafter</a> | ~8G |

```bash
pretrained/v2a/foleycrafter
├── checkpoints
    ├── auffusion
    ├── semantic
    ├── vocoder
    ├── temporal_adapter.ckpt
    └── timestamp_detector.pth.tar

```



## Remove voice-over

### BS Roformer
| Model    | Download Weights | Download Config | File size |
| -------- | ------- | ------- | ------- |
| BS Roformer | <a href="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"> Weights</a> | <a href="https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml"> Config</a> |610M |

BS Roformer demonstrated the best performance in our separation tasks, so we selected it as the exclusive model for voice-over removal in our experiments. Download model to `pretrained/remove_vo/checkpoints`
