"""Check native event data, quiet controls, clip timing and every encoded frame."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import imageio_ffmpeg

TRIAL=os.environ.get('BABYBOTTLE_TRIAL','heavy');assert TRIAL in ('light','heavy')
OUT=Path(f'/data/sriram/babybottle_thermal/{TRIAL}_multiview')
m=json.loads((OUT/'event_camera.json').read_text());d=np.load(OUT/'event_camera_events.npz')
assert m['trial']==TRIAL and m['scene']==('01_Light_Force' if TRIAL=='light' else '02_Heavy_Force')
assert m['camera']=='visible_24017931'
t,x,y,p=(d[k] for k in ['t_scene_s','x','y','polarity'])
assert len(t)==len(x)==len(y)==len(p)>0
assert np.isfinite(t).all() and np.all(np.diff(t)>=-1e-7)
start,end=m['scene_interval_s'];assert t.min()>=start and t.max()<=end+1e-6
assert x.max()<800 and y.max()<600 and set(np.unique(p))=={-1,1}
assert np.count_nonzero(p==1)==m['total_on'] and np.count_nonzero(p==-1)==m['total_off']
assert len(m['records'])==m['output_frames']==round(m['duration_s']*30)
assert m['input_frames']==m['output_frames']*4+1
assert sum(r['on'] for r in m['records'])==m['total_on']
assert sum(r['off'] for r in m['records'])==m['total_off']
assert np.allclose([r['scene_time_s'] for r in m['records']],start+(np.arange(m['output_frames'])+1)/30)
counts=np.array([r['on']+r['off'] for r in m['records']]);visible=np.array([r['hand_visible'] for r in m['records']])
assert visible.any() and not visible[0] and not visible[-1]
quiet=np.r_[counts[:10],counts[-10:]]
assert counts[visible].mean()>10*max(float(quiet.mean()),1.)
# Cycles/denoising is not bitwise stationary: the heavy tail contains two
# isolated ON events at (424,324) and (341,350), in separate video windows.
# Retain and report these artifacts rather than filtering the raw stream.
assert np.all(counts[:10]==0),'Unexpected events in stationary lead-in'
assert quiet.max()<=1 and quiet.sum()<=2,'Excess rendering artifacts in quiet windows'
assert hashlib.sha256(Path(m['source_blend']).read_bytes()).hexdigest()==m['source_sha256']
videos=[]
for name,size in [('event_camera.mp4',(800,680)),('event_rgb_comparison.mp4',(1600,680))]:
    reader=imageio_ffmpeg.read_frames(str(OUT/name));info=next(reader);count=0
    assert tuple(info['size'])==size and info['fps']==30
    for raw in reader:assert len(raw)==size[0]*size[1]*3;count+=1
    assert count==m['output_frames'] and abs(info['duration']-m['duration_s'])<.01
    videos.append(dict(file=name,frames=count,size=size,duration_s=info['duration']))
    print('VERIFIED',TRIAL,name,flush=True)
report=dict(passed=True,videos=videos,event_count=len(t),on=int((p==1).sum()),off=int((p==-1).sum()),
    quiet_mean_events_per_frame=float(quiet.mean()),quiet_total_events=int(quiet.sum()),
    quiet_max_events_per_frame=int(quiet.max()),hand_visible_mean_events_per_frame=float(counts[visible].mean()),
    source_unchanged=True,timestamps_and_coordinates_valid=True)
(OUT/'event_camera_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
