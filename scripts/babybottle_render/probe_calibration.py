"""Provisional Cam1 blackbody interpolation, NOT validated radiometric calibration.

Assumes directory suffixes are blackbody setpoints in Celsius. Camera drift,
blackbody emissivity, acquisition mode and object emissivity remain uncorrected.
"""
import json
import zipfile
from pathlib import Path
import numpy as np
from inspect_recordings import header

SOURCE = Path('/data/sriram/thermal_tactile_force/force_interaction_data/source')
OUT = Path('/data/sriram/babybottle_thermal/inspection/calibration_probe.json')


def read_frame(path, index):
    with zipfile.ZipFile(path) as archive, archive.open('raw_thr_frames.npy') as stream:
        shape, order, dtype = header(stream)
        assert not order
        size = int(np.prod(shape[1:])) * dtype.itemsize
        stream.seek(stream.tell() + index * size)
        return np.frombuffer(stream.read(size), dtype).reshape(shape[1:]).squeeze()[2:]


result = {'status': 'provisional apparent temperatures; uncorrected camera drift/emissivity',
          'camera': 'Cam1_274849', 'blackbody_roi_xyxy': [280, 80, 320, 110], 'blackbody': []}
for temperature in (20, 30, 40, 50):
    path = SOURCE / f'x_blackbody_{temperature}' / f'x_blackbody_{temperature}_Cam1_274849.npz'
    counts = [float(np.median(read_frame(path, i)[80:110, 280:320])) for i in (0, 150, 300, 450, 600, 750)]
    result['blackbody'].append({'assumed_C': temperature, 'sample_medians_counts': counts,
                               'median_counts': float(np.median(counts))})
xp = [v['median_counts'] for v in result['blackbody']]
assert np.all(np.diff(xp) > 0)
result['trials'] = {}
for trial in (1, 2):
    path = SOURCE / f'a_babybottle_{trial}' / f'a_babybottle_{trial}_Cam1_274849.npz'
    baseline = read_frame(path, 0)
    contact = read_frame(path, 180)
    # Broad image summaries only: no claim that these are segmented object temperatures.
    raw = [float(np.median(baseline)), float(np.percentile(contact, 95)), float(np.percentile(contact, 99))]
    result['trials'][str(trial)] = {'statistics': ['whole_image_precontact_median', 'contact_frame_180_p95', 'contact_frame_180_p99'],
                                   'raw_counts': raw, 'provisional_apparent_C': np.interp(raw, xp, [20, 30, 40, 50]).tolist()}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
