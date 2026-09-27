import hashlib,json
import numpy as np
import imageio_ffmpeg
from render_lidar_3d import OUT,View,TRIAL,range_colors

m=json.loads((OUT/'lidar_3d.json').read_text())
assert hashlib.sha256((OUT/'lidar_visible_returns.npz').read_bytes()).hexdigest()==m['source_sha256']
d=np.load(OUT/'lidar_visible_returns.npz')
assert m['color_source']=='sensor_radial_range_m' and m['color_limits_m']==[.25,1.25]
assert np.array_equal(range_colors([0,.25]),np.repeat(range_colors([.25]),2,axis=0))
assert np.array_equal(range_colors([1.25,2]),np.repeat(range_colors([1.25]),2,axis=0))
assert not np.array_equal(range_colors([.25]),range_colors([1.25]))
eye=np.array(m['viewer_eye_wide']);sensor=d['sensor_to_world'][:3,3]
assert np.linalg.norm(eye-sensor)>.1
view=View([1,-.75,.65],[.08,-.06,.01],48)
uv,z=view.project(np.array([[.08,-.06,.01]]));assert np.allclose(uv,[640,360]) and z[0]>0
assert len(m['reports'])==450
for i,r in enumerate(m['reports']):
    assert r['current_returns']==np.isfinite(d['range_m'][i]).sum()
    assert r['hand_returns']==(d['object_id'][i]==1).sum()
scan=json.loads((OUT/'lidar_visible.json').read_text())
assert scan['scene']==('01_Light_Force' if TRIAL=='light' else '02_Heavy_Force')
assert m.get('trial',TRIAL)==TRIAL
for row,source in zip(m['reports'],scan['frames_report']):
    assert row['hand_returns']==source['hand_returns']
    if not source['hand_visible']:assert row['hand_returns']==0
assert m['reports'][0]['hand_returns']==m['reports'][-1]['hand_returns']==0
assert sum(r['hand_returns'] for r in m['reports'])>1000
videos=[]
for name in m['videos']:
    reader=imageio_ffmpeg.read_frames(str(OUT/name));info=next(reader);n=0
    assert tuple(info['size'])==(1920,800) and info['fps']==30
    for frame in reader:assert len(frame)==1920*800*3;n+=1
    assert n==450
    videos.append(dict(file=name,frames=n,fps=info['fps'],size=info['size']))
    print('VERIFIED',name,flush=True)
(OUT/'lidar_3d_verification.json').write_text(json.dumps(dict(passed=True,videos=videos,
    checks='Full video decoding, unchanged source scan, independent third-person viewpoint, view-axis projection, per-frame return counts, hand appearance and disappearance.'),indent=2))
