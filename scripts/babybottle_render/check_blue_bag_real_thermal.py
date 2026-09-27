"""Verify recorded blue-bag video durations, geometry and normalization data."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial', required=True, choices=('light', 'heavy'))
    p.add_argument('--out-base', type=Path, default=Path('/data/sriram/blender_renders/s_blue_bag_in_hand'))
    a = p.parse_args()
    out = a.out_base / f'{a.trial}_multiview'
    m = json.loads((out/'real_thermal.json').read_text())
    assert m['trial'] == a.trial and m['frames'] == 900 and m['fps'] == 60
    assert len(m['camera_names']) == 4 and len(m['selected_source_frame_indices']) == 4
    assert max(m['max_timestamp_error_ms']) < 10
    ranges = np.load(out/'camera_normalization_ranges_counts.npy')
    assert ranges.shape == (900, 4, 2) and np.isfinite(ranges).all()
    assert np.all(ranges[..., 1] > ranges[..., 0])
    assert np.std(ranges[..., 0], axis=0).min() > 0
    videos = []
    for name, size in [(f'camera_{i}_real_normalized.mp4', (640, 560)) for i in range(4)] + [
        ('all_cameras_real_normalized.mp4', (1280, 1120))]:
        path = out/name
        reader = imageio_ffmpeg.read_frames(str(path))
        info = next(reader)
        assert tuple(info['size']) == size and info['fps'] == 60
        count = 0
        for frame in reader:
            assert len(frame) == size[0]*size[1]*3
            count += 1
        assert count == 900 and abs(info['duration']-15) < .01
        videos.append(dict(file=name, frames=count, size=size, duration_s=info['duration']))
    report = dict(passed=True, capture=m['capture'], videos=videos,
                  max_timestamp_error_ms=m['max_timestamp_error_ms'],
                  normalization_ranges_shape=list(ranges.shape))
    (out/'real_thermal_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
