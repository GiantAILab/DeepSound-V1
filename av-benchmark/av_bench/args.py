from argparse import ArgumentParser
from pathlib import Path


def get_eval_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # maximum length; does not pad if under (except internally in like PaSST)
    parser.add_argument('--audio_length', type=float)
    parser.add_argument('--num_workers', type=int, default=32)

    # only bs=1 supports variable audio length
    parser.add_argument('--gt_batch_size', type=int, default=1)
    # typically your generations would have the same length; so larger batch sizes can be used
    parser.add_argument('--pred_batch_size', type=int, default=1)

    parser.add_argument('--gt_audio', type=Path)
    parser.add_argument('--gt_cache', type=Path)
    parser.add_argument('--pred_audio', type=Path)
    parser.add_argument('--pred_cache', type=Path)

    parser.add_argument('--recompute_gt_cache', action='store_true')
    parser.add_argument('--recompute_pred_cache', action='store_true')

    # only single sample is supported
    # parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--unpaired', action='store_true')

    parser.add_argument('--skip_video_related',
                        action='store_true',
                        help='skips ImageBind and SynchFormer computation')
    parser.add_argument('--skip_clap', action='store_true', help='skips CLAP computation')

    parser.add_argument('--output_metrics_file', type=str, default='output_metrics.json')

    return parser
