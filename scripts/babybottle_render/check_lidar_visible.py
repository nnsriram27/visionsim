"""Validate projected LiDAR data and decode the complete RGB overlay movies."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import imageio_ffmpeg

TRIAL=os.environ.get('BABYBOTTLE_TRIAL','heavy')
assert TRIAL in ('light','heavy')
OUT=Path(f'/data/sriram/babybottle_thermal/{TRIAL}_multiview')
meta=json.loads((OUT/'lidar_visible.json').read_text());data=np.load(OUT/'lidar_visible_returns.npz')
ranges=data['range_m'];ideal=data['ideal_range_m'];uv=data['uv_pixels'];ids=data['object_id'];K=data['K']
assert ranges.shape==ideal.shape==ids.shape==(450,8000)
assert uv.shape==(450,8000,2)
valid=np.isfinite(ranges)
assert np.array_equal(valid,np.isfinite(ideal))
assert np.array_equal(valid,np.isfinite(uv).all(axis=-1))
assert np.all(ids[~valid]==-1) and set(np.unique(ids[valid])).issubset({0,1})
assert np.all((ranges[valid]>=.1)&(ranges[valid]<=450))
assert np.allclose(data['frame_time_s'],np.arange(450)/30)
assert np.allclose(data['firing_offset_s'],np.arange(8000)/240000)
assert data['firing_offset_s'][-1]<1/30
noise=ranges[valid]-ideal[valid]
assert abs(float(noise.mean()))<.0001
assert abs(float(noise.std())-.02)<.0001
maximum_projection_error=0
for i,record in enumerate(meta['frames_report']):
    index=i*8000+np.arange(8000,dtype=np.float64)
    yaw=np.deg2rad((np.mod(.5+index*.7548776662466927,1)-.5)*70.4)
    pitch=np.deg2rad((np.mod(.5+index*.5698402909980532,1)-.5)*77.2)
    expected=np.column_stack((K[0,2]+K[0,0]*np.tan(yaw),K[1,2]-K[1,1]*np.tan(pitch)))
    maximum_projection_error=max(maximum_projection_error,float(abs(expected[valid[i]]-uv[i,valid[i]]).max()))
    assert valid[i].sum()==record['returns']
    assert (ids[i]==1).sum()==record['hand_returns']
    if not record['hand_visible']:assert not (ids[i]==1).any()
    # Rendering clips float64 coordinates before the NPZ's float32 conversion.
    # A return within 3e-5 px of an image boundary can round across that boundary
    # in storage. Recompute the exact scan coordinates for the count check;
    # separately enforce the stored-coordinate error bound above.
    xy=np.floor(expected[valid[i]])
    visible=(xy[:,0]>=1)&(xy[:,0]<999)&(xy[:,1]>=1)&(xy[:,1]<749)
    assert visible.sum()==record['projected']
assert maximum_projection_error<.001
assert not np.allclose(uv[0],uv[1],equal_nan=True),'Scan repeats each frame'
assert np.count_nonzero(ids==1)>1000,'Moving hand not scanned'
assert meta['projection_error_px']<.002
assert hashlib.sha256(Path(meta['source_blend']).read_bytes()).hexdigest()==meta['source_sha256']
videos=[]
for name in ([] if meta.get('scan_only') else ['rgb','overlay','comparison']):
    path=OUT/f'lidar_visible_{name}.mp4';reader=imageio_ffmpeg.read_frames(str(path));info=next(reader);count=0
    expected_size=(2000,844) if name=='comparison' else (1000,844)
    assert tuple(info['size'])==expected_size
    for frame in reader:
        assert len(frame)==expected_size[0]*expected_size[1]*3;count+=1
    assert count==450 and info['fps']==30
    videos.append(dict(file=path.name,frames=count,size=info['size'],fps=info['fps']))
    print('VERIFIED',path.name,flush=True)
report=dict(passed=True,videos=videos,total_returns=int(valid.sum()),
    hand_returns=int(np.count_nonzero(ids==1)),projection_roundtrip_max_error_px=maximum_projection_error,
    blender_projection_max_error_px=meta['projection_error_px'],range_noise_mean_m=float(noise.mean()),
    range_noise_std_m=float(noise.std()),source_blend_unchanged=True,
    checks='Shapes, finite/missing returns, range limits, ray timestamps, non-repeated scan, animated hand presence, projection counts, camera intrinsics, range noise, source hash, all video frames decoded.')
(OUT/'lidar_visible_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
