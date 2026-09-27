"""Verify one object's light/heavy camera-1 and visible exports."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def verify(out, capture, trial):
    videos=[]
    for mode,name,shape,size,fps,frames in [
        ('thermal','camera_1_real_normalized.mp4',(900,2),(640,560),60,900),
        ('visible','visible_real_normalized.mp4',(225,2),(1920,1440),15,225)]:
        m=json.loads((out/f'{mode}_real_normalized.json').read_text())
        assert m['capture']==capture and m['trial']==trial
        assert m['frames']==frames and m['fps']==fps and m['file']==name
        assert m['source_frames']>=180 if mode=='visible' else m['source_frames']>=900
        assert Path(m['source']).exists()
        selected=np.asarray(m['source_frame_indices'])
        assert len(selected)==frames and np.all(np.diff(selected)>=0)
        assert selected.min()>=0 and selected.max()<m['source_frames']
        assert m['max_timing_error_ms']<(75 if mode=='visible' else 11)
        data=np.load(out/('camera_1_normalization_ranges_counts.npy' if mode=='thermal' else 'visible_normalization_ranges_counts.npy'))
        assert data.shape==shape and np.isfinite(data).all()
        assert np.all(data[:,1]>data[:,0])
        reader=imageio_ffmpeg.read_frames(str(out/name));info=next(reader);count=0
        assert tuple(info['size'])==size and info['fps']==fps
        for raw in reader:
            assert len(raw)==size[0]*size[1]*3
            count+=1
        assert count==frames and abs(info['duration']-15)<.01
        videos.append(dict(file=name,frames=count,fps=fps,size=size,max_timing_error_ms=m['max_timing_error_ms']))
    result=dict(passed=True,capture=capture,trial=trial,videos=videos)
    (out/'real_capture_verification.json').write_text(json.dumps(result,indent=2))
    print(capture,'VERIFIED',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',required=True)
    p.add_argument('--output-root',type=Path,default=Path('/data/sriram/blender_renders'))
    a=p.parse_args()
    for suffix,trial in [('1','light'),('2','heavy')]:
        verify(a.output_root/a.object/f'{trial}_multiview',f'{a.object}_{suffix}',trial)


if __name__=='__main__':main()
