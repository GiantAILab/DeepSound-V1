import json
import logging
from pathlib import Path

import torch
from colorlog import ColoredFormatter

from av_bench.args import get_eval_parser
from av_bench.evaluate import evaluate
from av_bench.extract import extract
from argparse import ArgumentParser

log = logging.getLogger()
device = 'cuda'

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


setup_eval_logging()


@torch.inference_mode()
def main(args):
    gt_audio: Path = args.gt_audio
    gt_cache: Path = args.gt_cache
    pred_audio: Path = args.pred_audio
    pred_cache: Path = args.pred_cache
    audio_length: float = args.audio_length
    num_workers: int = args.num_workers
    gt_batch_size: int = args.gt_batch_size
    pred_batch_size: int = args.pred_batch_size
    unpaired: bool = args.unpaired
    skip_video_related: bool = args.skip_video_related
    skip_clap: bool = args.skip_clap
    recompute_gt_cache: bool = args.recompute_gt_cache
    recompute_pred_cache: bool = args.recompute_pred_cache
    

    gt_cache = gt_cache.expanduser()
    pred_cache = pred_cache.expanduser()

    log.info(f'GT cache path: {gt_cache}')
    log.info(f'Pred cache: {pred_cache}')
    log.info(f'Audio length: {audio_length}')
    log.info(f'Unpaired: {unpaired}')

    # extract for GT if needed
    if not (gt_cache / 'passt_features_embed.pth').exists() or recompute_gt_cache:
        log.info('Extracting GT cache...')
        if gt_audio is None:
            raise ValueError('Must specify gt_audio to compute gt_cache')
        gt_audio = gt_audio.expanduser()
        log.info(f'GT audio path: {gt_audio}')
        extract(
            audio_path=gt_audio,
            output_path=gt_cache,
            audio_length=audio_length,
            device=device,
            batch_size=gt_batch_size,
            num_workers=num_workers,
            skip_video_related=True,
            skip_clap=True,
        )

    # extract for pred if needed
    if not (pred_cache / 'passt_features_embed.pth').exists() or recompute_pred_cache:
        log.info('Extracting pred cache...')
        if pred_audio is None:
            raise ValueError('Must specify pred_audio to compute pred_cache')
        pred_audio = pred_audio.expanduser()
        log.info(f'Pred audio path: {pred_audio}')
        extract(
            audio_path=pred_audio,
            output_path=pred_cache,
            audio_length=audio_length,
            device=device,
            batch_size=pred_batch_size,
            num_workers=num_workers,
            skip_video_related=skip_video_related,
            skip_clap=skip_clap,
        )
    

    log.info('Starting evaluation...')

    num_samples = 1
    output_metrics = evaluate(gt_audio_cache=gt_cache,
                              pred_audio_cache=pred_cache,
                              num_samples=num_samples,
                              is_paired=not unpaired)

    for k, v in output_metrics.items():
        # pad k to 10 characters
        # pad v to 10 decimal places
        log.info(f'{k:<10}: {v:.10f}')

    # write output metrics to file
    # output_metrics_file = pred_cache / 'output_metrics.json'
    output_metrics_file = args.output_metrics_file
    with open(output_metrics_file, 'w') as f:
        json.dump(output_metrics, f, indent=4)
    log.info(f'Output metrics written to {output_metrics_file}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = get_eval_parser().parse_args()
    main(args)
