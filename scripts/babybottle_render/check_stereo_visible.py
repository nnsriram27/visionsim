"""Decode stereo deliverables and verify metric disparity, timing and rig geometry."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import imageio_ffmpeg

trial=os.environ.get('BABYBOTTLE_TRIAL','heavy')
out=Path(f'/data/sriram/babybottle_thermal/{trial}_multiview')
m=json.loads((out/'stereo.json').read_text());d=np.load(out/'stereo_disparity.npz')
left=d['left_to_world'];right=d['right_to_world'];center=np.array(m['center_to_world'])
assert np.allclose(left[:3,:3],right[:3,:3])
assert np.allclose((left[:3,3]+right[:3,3])/2,center[:3,3])
assert abs(np.linalg.norm(left[:3,3]-right[:3,3])*m['unit_scale_m']-.065)<1e-7
assert m['projection_check_max_error_px']<.002
disp=d['disparity_px'];assert disp.shape==(m['frames'],600,800)
# Background above the table has no mesh intersection and is intentionally NaN.
assert np.isfinite(disp).mean()>.8 and np.nanmin(disp)>0
assert np.allclose(d['frame_time_s'],m['scene_interval_s'][0]+(np.arange(m['frames'])+1)/30)
assert all(r['eye_mean_absolute_difference']>0 for r in m['records'])
assert any(r['hand_visible'] for r in m['records']) and not m['records'][-1]['hand_visible']
assert hashlib.sha256(Path(m['source_blend']).read_bytes()).hexdigest()==m['source_sha256']
videos=[]
for name,width in [('stereo_sbs',1600),('stereo_disparity_turbo',800),('stereo_rgb_disparity',1600)]:
    reader=imageio_ffmpeg.read_frames(str(out/f'{name}.mp4'));info=next(reader);n=0
    assert tuple(info['size'])==(width,680) and info['fps']==30
    for raw in reader:assert len(raw)==width*680*3;n+=1
    assert n==m['frames'] and abs(info['duration']-m['duration_s'])<.01
    videos.append(dict(file=name+'.mp4',frames=n,duration_s=info['duration']))
report=dict(passed=True,source_unchanged=True,parallel_axes=True,baseline_m=.065,
    projection_error_px=m['projection_check_max_error_px'],finite_disparity_fraction=float(np.isfinite(disp).mean()),videos=videos)
(out/'stereo_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
