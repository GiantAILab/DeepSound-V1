# coding=utf-8
# V2A


def mmaudio_infer(args):
    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    # if args.video:
    #     video_path: Path = Path(args.video).expanduser()
    # else:
    #     video_path = None
    
    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    output_dir: str = args.output.expanduser()
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        log.warning('CUDA/MPS are not available, running on CPU')
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    net: MMAudio = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()
    
    video_paths = []
    # raw_video_dir = "/ailab-train/speech/zhanghaomin/VGGSound/video"
    with open(args.test_file, "r", encoding="utf-8") as fr:
        for line in tqdm(fr.readlines()[args.start_idx: args.end_idx]):
            # video_name = line.strip().split("\t")[0].split("/")[-1]
            # vpath = f"{raw_video_dir}/{video_name}"
            vpath = line.strip().split("\t")[0]
            vpath = Path(vpath).expanduser()
            video_paths.append(vpath)
            
    st = time.time()
    for video_path in tqdm(video_paths):
        duration = args.duration

        if video_path is not None:
            log.info(f'Using video {video_path}')
            video_info = load_video(video_path, duration)
            clip_frames = video_info.clip_frames
            sync_frames = video_info.sync_frames
            duration = video_info.duration_sec
            if mask_away_clip:
                clip_frames = None
            else:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)
        else:
            log.info('No video provided -- text-to-audio mode')
            clip_frames = sync_frames = None

        seq_cfg.duration = duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

        log.info(f'Prompt: {prompt}')
        log.info(f'Negative prompt: {negative_prompt}')
        
        try:
            audios = generate(clip_frames,
                            sync_frames, [prompt],
                            negative_text=[negative_prompt],
                            feature_utils=feature_utils,
                            net=net,
                            fm=fm,
                            rng=rng,
                            cfg_strength=cfg_strength)
            audio = audios.float().cpu()[0]
            if video_path is not None:
                save_path = output_dir / f'{video_path.stem}.wav'
            else:
                safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
                save_path = output_dir / f'{safe_filename}.wav'
            torchaudio.save(str(save_path), audio, seq_cfg.sampling_rate)

            log.info(f'Audio saved to {save_path}')
            if video_path is not None and not skip_video_composite:
                video_save_path = output_dir / f'{video_path.stem}.mp4'
                make_video(video_info, str(video_save_path), audio, sampling_rate=seq_cfg.sampling_rate)
                log.info(f'Video saved to {output_dir / video_save_path}')

            log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
        except Exception as e:
            print(f"eeeeeeeeeeee on {video_path} for error {e}")
    et = time.time()
    print("ttttttttttttttttttt: ", et-st)


def foleycrafter_infer():
    pass



class Step1:
    def __init__(self, audio_generator):
        self.audio_generator = audio_generator

    def run(self, step1_info):
        print("[Step1] 生成音频和合成视频中...")
        generated_audio, video_with_audio = self.audio_generator.generate(step1_info)
        print("[Step1] 音频和视频生成完成")
        return generated_audio, video_with_audio
