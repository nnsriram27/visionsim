"""Export the eight recorded blue-bag thermal streams as normalized Inferno MP4s.

The source arrays are uncompressed .npy members inside .npz archives. Seek to
selected frames without loading the full 565 MB archive into memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
from numpy.lib import format as npy_format
from PIL import Image, ImageDraw, ImageFont

SOURCE = Path('/data/sriram/thermal_tactile_force/force_interaction_data/source')
DEFAULT_OUTPUT = Path('/data/sriram/blender_renders/s_blue_bag_in_hand')
CAMERAS = ('Cam0_268771', 'Cam1_274849', 'Cam2_275171', 'Cam3_60393')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FPS = 60
HEIGHT, WIDTH = 512, 640
PANEL_HEIGHT = 560


class RecordedCamera:
    def __init__(self, path: Path, stack: ExitStack):
        self.path = path
        self.archive = stack.enter_context(zipfile.ZipFile(path))
        self.timestamps = np.load(self.archive.open('raw_thr_tstamps.npy'))
        self.stream = stack.enter_context(self.archive.open('raw_thr_frames.npy'))
        version = npy_format.read_magic(self.stream)
        assert version == (1, 0), (path, version)
        self.shape, order, self.dtype = npy_format.read_array_header_1_0(
            self.stream, max_header_size=30000)
        assert self.shape == (len(self.timestamps), 514, WIDTH, 1), self.shape
        assert not order and self.dtype == np.dtype('uint16')
        self.data_offset = self.stream.tell()
        self.frame_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        assert np.all(np.diff(self.timestamps) > 0)

    def frame(self, index: int) -> np.ndarray:
        self.stream.seek(self.data_offset + index * self.frame_bytes)
        raw = self.stream.read(self.frame_bytes)
        if len(raw) != self.frame_bytes:
            raise EOFError(f'{self.path}: truncated frame {index}')
        # The first two rows are FLIR telemetry, as in the existing bottle pipeline.
        return np.frombuffer(raw, self.dtype).reshape(514, WIDTH)[2:]


def writer(path: Path, width: int, height: int) -> subprocess.Popen:
    return subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
        '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '18', '-threads', '2', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path),
    ], stdin=subprocess.PIPE)


def panel(frame: np.ndarray, camera: int, trial: str, t: float,
          lut: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    low, high = (float(v) for v in np.percentile(frame, (1, 99)))
    if high <= low:
        high = low + 1.0
    lookup = np.rint(np.clip((frame.astype(np.float32)-low)/(high-low), 0, 1)*2047).astype(np.uint16)
    im = Image.new('RGB', (WIDTH, PANEL_HEIGHT), (14, 18, 24))
    im.paste(Image.fromarray(lut[lookup]), (0, 40))
    draw = ImageDraw.Draw(im)
    draw.text((10, 8), f'CAMERA {camera} | REAL | {trial.upper()} | INFERNO P1-P99',
              font=ImageFont.truetype(FONT, 17), fill='white')
    draw.text((10, 531), f'{t:5.2f} s  |  frame scale {low:.0f} to {high:.0f} raw counts',
              font=ImageFont.truetype(FONT, 15), fill='#d4dce5')
    bar = lut[np.rint(np.linspace(0, 2047, 120)).astype(int)]
    im.paste(Image.fromarray(np.repeat(bar[None], 12, axis=0)), (508, 542))
    return np.asarray(im), (low, high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--trial', choices=('light', 'heavy'), required=True)
    parser.add_argument('--out-base', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--frames', type=int, default=900,
                        help='Normally 900 frames (15 s); smaller values permit a preview run')
    args = parser.parse_args()
    trial_number = 1 if args.trial == 'light' else 2
    input_dir = SOURCE / f's_blue_bag_in_hand_{trial_number}'
    out = args.out_base / f'{args.trial}_multiview'
    out.mkdir(parents=True, exist_ok=True)
    lut = (matplotlib.colormaps['inferno'](np.linspace(0, 1, 2048))[:, :3]*255).astype(np.uint8)
    started = time.monotonic()

    with ExitStack() as stack:
        cameras = [RecordedCamera(input_dir / f's_blue_bag_in_hand_{trial_number}_{name}.npz', stack)
                   for name in CAMERAS]
        assert all(len(c.timestamps) >= args.frames for c in cameras)
        assert args.frames > 0 and args.frames <= 900
        # Timestamps are already on a common clock. A median start avoids choosing
        # one camera as master; each output frame selects the nearest capture.
        t0 = float(np.median([c.timestamps[0] for c in cameras]))
        target = t0 + np.arange(args.frames)/FPS
        indices = []
        for c in cameras:
            right = np.searchsorted(c.timestamps, target)
            right = np.clip(right, 0, len(c.timestamps)-1)
            left = np.maximum(right-1, 0)
            selected = np.where(np.abs(c.timestamps[left]-target) <=
                                np.abs(c.timestamps[right]-target), left, right)
            errors = np.abs(c.timestamps[selected]-target)
            assert errors.max() < 0.010, (c.path, errors.max())
            assert np.all(np.diff(selected) >= 0)
            indices.append(selected)

        writers = [writer(out/f'camera_{i}_real_normalized.mp4', WIDTH, PANEL_HEIGHT)
                   for i in range(4)]
        overview = writer(out/'all_cameras_real_normalized.mp4', WIDTH*2, PANEL_HEIGHT*2)
        ranges = np.empty((args.frames, 4, 2), np.float32)
        for frame_number in range(args.frames):
            tiles = []
            for camera_number, camera in enumerate(cameras):
                raw = camera.frame(int(indices[camera_number][frame_number]))
                tile, bounds = panel(raw, camera_number, args.trial, frame_number/FPS, lut)
                ranges[frame_number, camera_number] = bounds
                writers[camera_number].stdin.write(tile.tobytes())
                tiles.append(tile)
            grid = np.concatenate((np.concatenate(tiles[:2], axis=1),
                                   np.concatenate(tiles[2:], axis=1)), axis=0)
            overview.stdin.write(grid.tobytes())
            if frame_number in (0, 180, 450, args.frames-1):
                Image.fromarray(grid).save(out/f'real_preview_{frame_number:04d}.jpg', quality=95)
            if frame_number % 60 == 0:
                print(args.trial, frame_number, '/', args.frames,
                      'elapsed', round(time.monotonic()-started, 1), flush=True)
        for process in [*writers, overview]:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('ffmpeg encoding failed')

        np.save(out/'camera_normalization_ranges_counts.npy', ranges)
        metadata = dict(
            capture=f's_blue_bag_in_hand_{trial_number}', trial=args.trial,
            source_files=[str(c.path) for c in cameras],
            camera_names=list(CAMERAS), source_frame_counts=[len(c.timestamps) for c in cameras],
            frames=args.frames, fps=FPS, duration_s=args.frames/FPS,
            sensor_size=[WIDTH, HEIGHT], individual_video_size=[WIDTH, PANEL_HEIGHT],
            overview_video_size=[WIDTH*2, PANEL_HEIGHT*2],
            target_start_timestamp_s=t0,
            selected_source_frame_indices=[v.tolist() for v in indices],
            max_timestamp_error_ms=[float(np.max(np.abs(c.timestamps[v]-target))*1000)
                                    for c, v in zip(cameras, indices)],
            normalization='Independent per-camera, per-frame P1-P99 of the 640x512 recorded raw counts; clip to bounds, then Inferno. No temporal smoothing or temperature calibration.',
            data='Only recorded raw thermal frames. First two telemetry rows omitted. No simulation, denoising, or synthetic noise.',
            outputs=[f'camera_{i}_real_normalized.mp4' for i in range(4)] +
                    ['all_cameras_real_normalized.mp4', 'camera_normalization_ranges_counts.npy'],
            elapsed_s=time.monotonic()-started,
        )
        (out/'real_thermal.json').write_text(json.dumps(metadata, indent=2))
        (out/'README.md').write_text(
            f'# {args.trial.title()} blue-bag real thermal capture\n\n'
            f'15 seconds at 60 fps, cameras 0–3. Per-frame, per-camera P1–P99 '
            f'normalization of the recorded raw counts, shown with Inferno. '
            f'The scale changes per frame; color does not represent a fixed temperature.\n\n'
            + ''.join(f'- [Camera {i}](camera_{i}_real_normalized.mp4)\n' for i in range(4))
            + '- [All four views](all_cameras_real_normalized.mp4)\n\n'
            + 'Capture timestamps align the four views. `real_thermal.json` contains '
            + 'source paths and selected frame indices; '
            + '`camera_normalization_ranges_counts.npy` contains the displayed raw-count bounds.\n'
        )
    print('REAL THERMAL COMPLETE', args.trial, flush=True)


if __name__ == '__main__':
    main()
