"""Debayer and encode the recorded blue-bag visible-camera clips."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from numpy.lib import format as npy_format
from PIL import Image

SOURCE = Path('/data/sriram/thermal_tactile_force/force_interaction_data/source')
OUTPUT = Path('/data/sriram/blender_renders/s_blue_bag_in_hand')
FFMPEG = Path('/data/sriram/babybottle_thermal/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
SOURCE_H, SOURCE_W = 3000, 4000
OUT_W, OUT_H = 1920, 1440
FPS, DURATION_S = 15, 15


def encoder(path):
    return subprocess.Popen([
        str(FFMPEG), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt',
        'rgb24', '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-threads', '4',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
    ], stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', required=True, choices=('light', 'heavy'))
    parser.add_argument('--frames', type=int, default=FPS*DURATION_S)
    parser.add_argument('--out-base', type=Path, default=OUTPUT)
    args = parser.parse_args()
    assert 1 <= args.frames <= FPS*DURATION_S
    capture = f's_blue_bag_in_hand_{1 if args.trial == "light" else 2}'
    source = SOURCE/capture/f'{capture}_vis.npz'
    out = args.out_base/f'{args.trial}_multiview'
    assert out.is_dir()
    start = time.monotonic()
    with zipfile.ZipFile(source) as archive, archive.open('raw_vis_frames.npy') as stream:
        stamps = np.load(archive.open('vis_cam_tstamps.npy'))
        unix = np.load(archive.open('vis_unix_tstamps.npy'))
        frame_ids = np.load(archive.open('vis_frame_ids.npy'))
        exposure = np.load(archive.open('vis_exposure_us.npy'))
        gain = np.load(archive.open('vis_gain_db.npy'))
        pixel_format = str(np.load(archive.open('vis_pixel_format.npy')).item())
        camera = str(np.load(archive.open('vis_serial.npy')).item())
        model = str(np.load(archive.open('vis_model.npy')).item())
        assert pixel_format == 'BayerRG16' and camera == '24017931'
        version = npy_format.read_magic(stream)
        assert version == (1, 0)
        shape, order, dtype = npy_format.read_array_header_1_0(stream, max_header_size=30000)
        assert shape == (180, SOURCE_H, SOURCE_W) and not order and dtype == np.uint16
        assert len(stamps) == len(unix) == len(frame_ids) == 180
        assert np.all(np.diff(stamps) > 0) and np.all(np.diff(frame_ids) > 0)
        offset = stream.tell()
        frame_bytes = SOURCE_H*SOURCE_W*2
        target = stamps[0]+np.arange(args.frames)/FPS
        right = np.clip(np.searchsorted(stamps, target), 0, len(stamps)-1)
        left = np.maximum(right-1, 0)
        selection = np.where(abs(stamps[left]-target) <= abs(stamps[right]-target), left, right)
        assert np.all(np.diff(selection) >= 0)
        if args.frames == FPS*DURATION_S:
            assert np.max(abs(stamps[selection]-target)) < .075

        # Balance on the printed white calibration board in the first frame.
        # The camera stores raw Bayer values without display white balance.
        stream.seek(offset)
        first_bayer = np.frombuffer(stream.read(frame_bytes), dtype=np.uint16).reshape(SOURCE_H, SOURCE_W)
        first_rgb16 = cv2.cvtColor(first_bayer, cv2.COLOR_BayerRG2BGR)
        board = cv2.resize(first_rgb16, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)[500:1000, 600:1250]
        white_pixels = board[(board.min(axis=2)>5000)&(board.max(axis=2)<50000)]
        assert len(white_pixels)>20000, len(white_pixels)
        white_medians = np.median(white_pixels, axis=0)
        gains = white_medians[1]/white_medians
        assert np.all((gains>.5)&(gains<3)), gains
        # One stable display transform for the full capture.
        black, white, gamma = 64., 45000., .55
        lookup = np.rint(np.clip((np.arange(65536)-black)/(white-black), 0, 1)**gamma*255).astype(np.uint8)

        def image(index):
            if index == 0:
                rgb16 = first_rgb16
            else:
                stream.seek(offset+int(index)*frame_bytes)
                data = stream.read(frame_bytes)
                if len(data) != frame_bytes:
                    raise EOFError(f'Truncated visible frame {index}')
                bayer = np.frombuffer(data, dtype=np.uint16).reshape(SOURCE_H, SOURCE_W)
                # OpenCV's BayerRG2BGR output has R in array channel 0 for this
                # RGGB mosaic. Verified with a synthetic R/G/B Bayer pattern.
                rgb16 = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
            rgb8 = lookup[np.clip(rgb16.astype(np.float32)*gains, 0, 65535).astype(np.uint16)]
            return cv2.resize(rgb8, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

        process = encoder(out/'visible_real.mp4')
        last_index = -1
        frame = None
        for i, source_index in enumerate(selection):
            if source_index != last_index:
                frame = image(source_index)
                last_index = source_index
            process.stdin.write(frame.tobytes())
            if i in (0, 45, 90, 135, args.frames-1):
                Image.fromarray(frame).save(out/f'visible_preview_{i:04d}.jpg', quality=95)
            if i % 30 == 0:
                print(args.trial, i, '/', args.frames, 'elapsed', round(time.monotonic()-start, 1), flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed')

        metadata = dict(capture=capture, trial=args.trial, source=str(source),
            camera=camera, model=model, pixel_format=pixel_format,
            source_resolution=[SOURCE_W, SOURCE_H], output_resolution=[OUT_W, OUT_H],
            source_frames=len(stamps), encoded_frames=args.frames, fps=FPS,
            duration_s=args.frames/FPS, source_elapsed_s=float(stamps[-1]-stamps[0]),
            source_timestamps_s=stamps.tolist(), source_unix_timestamps_s=unix.tolist(),
            source_frame_ids=frame_ids.tolist(), selected_source_frame_indices=selection.tolist(),
            selected_timing_error_ms=(np.abs(stamps[selection]-target)*1000).tolist(),
            max_timing_error_ms=float(np.max(abs(stamps[selection]-target))*1000),
            skipped_source_frame_ids=int(frame_ids[-1]-frame_ids[0]+1-len(frame_ids)),
            exposure_us_min_max=[float(exposure.min()), float(exposure.max())],
            gain_db_min_max=[float(gain.min()), float(gain.max())],
            demosaic='OpenCV COLOR_BayerRG2BGR at native resolution; resulting array is RGB for top-left-red RGGB input, verified with a synthetic Bayer pattern',
            display_transform=dict(black_raw=black, white_raw=white, gamma=gamma,
                white_balance_reference='Bright pixels of printed white calibration board in first frame',
                reference_raw_channel_medians=white_medians.tolist(),rgb_channel_gains=gains.tolist(),
                applied='Fixed white balance and LUT on all native-resolution RGB frames; then area resize'),
            timing='Nearest recorded camera timestamp on a 15 fps, 15-second timeline; missing capture frames are held, never interpolated.',
            elapsed_s=time.monotonic()-start)
        (out/'visible_real.json').write_text(json.dumps(metadata, indent=2))
        readme = out/'README.md'
        note = ('\n## Recorded visible camera\n\n'
                '[Visible RGB video](visible_real.mp4) from camera 24017931, '
                '15 seconds at 15 fps, 1920×1440. Native 4000×3000 BayerRG16 '
                'frames are demosaiced, white balanced on the board in the first frame, '
                'given one fixed display transform, and '
                'downsampled for MP4. Capture timestamps govern playback; skipped '
                'source frames are held. See [metadata](visible_real.json).\n')
        existing = readme.read_text() if readme.exists() else f'# {args.trial.title()} blue-bag capture\n'
        if '## Recorded visible camera' not in existing:
            readme.write_text(existing+note)
    print('VISIBLE COMPLETE', args.trial, flush=True)


if __name__ == '__main__':
    main()
