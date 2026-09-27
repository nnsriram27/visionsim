"""Validate all frames and shared-scale metadata of one camera-1 comparison."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--object',required=True)
    p.add_argument('--output-root',type=Path,default=Path('/data/sriram/blender_renders'))
    a=p.parse_args();out=a.output_root/a.object
    m=json.loads((out/'camera_1_light_vs_heavy_global.json').read_text())
    assert m['object']==a.object and m['camera']=='Cam1_274849'
    assert m['frames']==900 and m['fps']==60 and m['duration_s']==15
    assert m['histogram_pixels']==2*(900-8*60)*640*512
    assert m['normalization_window_s']==[8.,15.] and m['upper_percentile']==99.9
    lo,hi=m['global_limits_raw_counts'];assert 0<=lo<hi<=65535
    assert m['color_map']=='inferno'
    for side,path in enumerate(m['source_files']):
        assert Path(path).is_file()
        selected=np.asarray(m['selected_source_frame_indices'][side])
        assert len(selected)==900 and selected.min()>=0
        assert selected.max()<m['source_frame_counts'][side]
        assert np.all(np.diff(selected)>=0)
        assert m['max_timestamp_error_ms'][side]<11
    video=out/'camera_1_light_vs_heavy_global.mp4'
    reader=imageio_ffmpeg.read_frames(str(video));info=next(reader);count=0
    assert tuple(info['size'])==(1280,592) and info['fps']==60
    for raw in reader:
        assert len(raw)==1280*592*3
        count+=1
    assert count==900 and abs(info['duration']-15)<.01
    result=dict(passed=True,object=a.object,file=video.name,frames=count,fps=60,
                duration_s=info['duration'],size=[1280,592],global_limits_raw_counts=[lo,hi],
                max_timestamp_error_ms=m['max_timestamp_error_ms'])
    (out/'camera_1_light_vs_heavy_global_verification.json').write_text(json.dumps(result,indent=2))
    print(a.object,'VERIFIED',lo,hi,flush=True)


if __name__=='__main__':main()
