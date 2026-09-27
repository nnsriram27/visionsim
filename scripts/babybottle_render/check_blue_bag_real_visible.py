"""Verify blue-bag visible export frame timing and video decoding."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trial',choices=('light','heavy'),required=True)
    p.add_argument('--out-base',type=Path,default=Path('/data/sriram/blender_renders/s_blue_bag_in_hand'))
    a=p.parse_args()
    out=a.out_base/f'{a.trial}_multiview'
    m=json.loads((out/'visible_real.json').read_text())
    assert m['trial']==a.trial and m['camera']=='24017931'
    assert m['source_frames']==180 and m['encoded_frames']==225 and m['fps']==15
    assert m['pixel_format']=='BayerRG16' and m['skipped_source_frame_ids']==33
    index=np.asarray(m['selected_source_frame_indices'])
    stamps=np.asarray(m['source_timestamps_s'])
    assert len(index)==225 and np.all(np.diff(index)>=0)
    assert np.all(index>=0) and np.all(index<180)
    assert np.allclose(np.abs(stamps[index]-(stamps[0]+np.arange(225)/15))*1000,
                       m['selected_timing_error_ms'],atol=.001)
    assert m['max_timing_error_ms']<75
    assert (out/'real_thermal_verification.json').exists()
    reader=imageio_ffmpeg.read_frames(str(out/'visible_real.mp4'))
    info=next(reader);size=(1920,1440);count=0
    assert tuple(info['size'])==size and info['fps']==15
    for frame in reader:
        assert len(frame)==size[0]*size[1]*3
        count+=1
    assert count==225 and abs(info['duration']-15)<.01
    report=dict(passed=True,trial=a.trial,video='visible_real.mp4',frames=count,
                fps=15,duration_s=info['duration'],size=size,
                source_frames=180,skipped_source_frame_ids=33,
                max_timing_error_ms=m['max_timing_error_ms'])
    (out/'visible_real_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
