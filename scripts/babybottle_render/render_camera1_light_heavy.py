"""Side-by-side real camera-1 thermals with one post-contact pair-wide scale."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import matplotlib
import numpy as np
from numpy.lib import format as npy_format
from PIL import Image, ImageDraw, ImageFont

SOURCE_ROOT = Path('/data/sriram/thermal_tactile_force/force_interaction_data/source')
OUTPUT_ROOT = Path('/data/sriram/blender_renders')
FFMPEG = Path('/data/sriram/babybottle_thermal/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
W,H,FPS,N = 640,512,60,900
PANEL_H=592


class RawThermal:
    def __init__(self,path:Path,stack:ExitStack):
        self.path=path
        self.archive=stack.enter_context(zipfile.ZipFile(path))
        self.stream=stack.enter_context(self.archive.open('raw_thr_frames.npy'))
        stamps=np.load(self.archive.open('raw_thr_tstamps.npy'))
        self.timestamps=stamps
        assert npy_format.read_magic(self.stream)==(1,0)
        shape,order,dtype=npy_format.read_array_header_1_0(self.stream,max_header_size=30000)
        assert not order and dtype==np.uint16 and shape[1:]==(514,W,1)
        assert len(stamps)==shape[0] and len(stamps)>=N
        self.shape=shape
        self.offset=self.stream.tell()
        target=stamps[0]+np.arange(N)/FPS
        right=np.clip(np.searchsorted(stamps,target),0,len(stamps)-1)
        left=np.maximum(right-1,0)
        self.selected=np.where(abs(stamps[left]-target)<=abs(stamps[right]-target),left,right)
        self.timing_error_ms=np.abs(stamps[self.selected]-target)*1000
        assert self.timing_error_ms.max()<11
        assert np.all(np.diff(self.selected)>=0)

    def image(self,output_index:int)->np.ndarray:
        source_index=int(self.selected[output_index])
        frame_bytes=514*W*2
        self.stream.seek(self.offset+source_index*frame_bytes)
        data=self.stream.read(frame_bytes)
        if len(data)!=frame_bytes:raise EOFError((self.path,source_index))
        return np.frombuffer(data,np.uint16).reshape(514,W)[2:]


def encoder(path:Path):
    return subprocess.Popen([str(FFMPEG),'-y','-loglevel','error','-f','rawvideo',
        '-pix_fmt','rgb24','-s',f'{2*W}x{PANEL_H}','-r',str(FPS),'-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','18','-threads','4',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)


def panel(raw,label,t,lo,hi,lut,bar):
    idx=np.rint(np.clip((raw.astype(np.float32)-lo)/(hi-lo),0,1)*2047).astype(np.uint16)
    im=Image.new('RGB',(W,PANEL_H),(14,18,24))
    im.paste(Image.fromarray(lut[idx]),(0,40))
    draw=ImageDraw.Draw(im)
    draw.text((12,9),f'CAMERA 1 | REAL | {label}',font=ImageFont.truetype(FONT,20),fill='white')
    draw.text((535,12),f'{t:.2f} s',font=ImageFont.truetype(FONT,15),fill='#d4dce5')
    draw.text((12,562),f'POST-HAND SCALE: {lo} to {hi} raw counts | INFERNO',
        font=ImageFont.truetype(FONT,14),fill='#d4dce5')
    im.paste(Image.fromarray(bar),(476,577))
    return np.asarray(im)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--object',required=True,help='For example a_tennis_ball or s_blue_bag_in_hand')
    p.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    p.add_argument('--output-root',type=Path,default=OUTPUT_ROOT)
    p.add_argument('--preview',action='store_true',help='Render a few hand and post-contact frames using the candidate scale')
    p.add_argument('--upper-percentile',type=float,default=99.9)
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args()
    assert 95<=args.upper_percentile<=99.9
    out=args.output_root/args.object
    assert out.is_dir()
    names=[f'{args.object}_1',f'{args.object}_2']
    paths=[args.source_root/name/f'{name}_Cam1_274849.npz' for name in names]
    start=time.monotonic()
    with ExitStack() as stack:
        streams=[RawThermal(path,stack) for path in paths]
        hist=np.zeros(65536,np.int64)
        post_start=8*FPS
        for side,stream in enumerate(streams):
            for i in range(post_start,N):
                hist+=np.bincount(stream.image(i).ravel(),minlength=65536)
            print(args.object,'histogram',side+1,'/ 2',round(time.monotonic()-start,1),flush=True)
        total=int(hist.sum())
        assert total==2*(N-post_start)*W*H
        cumulative=np.cumsum(hist)
        quantiles={str(q):int(np.searchsorted(cumulative,np.ceil(q/100*total),side='left'))
                   for q in (1,50,95,99,99.5,99.9)}
        print(args.object,'post-hand quantiles',quantiles,flush=True)
        if args.analyze_only:return
        lo=int(np.searchsorted(cumulative,np.ceil(.01*total),side='left'))
        hi=int(np.searchsorted(cumulative,np.ceil(args.upper_percentile/100*total),side='left'))
        assert hi>lo,(lo,hi)
        lut=(matplotlib.colormaps['inferno'](np.linspace(0,1,2048))[:,:3]*255).astype(np.uint8)
        bar=np.repeat(lut[np.rint(np.linspace(0,2047,150)).astype(int)][None],10,axis=0)
        video=out/'camera_1_light_vs_heavy_global.mp4'
        pending=out/'camera_1_light_vs_heavy_global.rendering.mp4'
        indices=[180,450,N-1] if args.preview else range(N)
        process=None if args.preview else encoder(pending)
        for i in indices:
            t=i/FPS
            frames=[panel(s.image(i),name,t,lo,hi,lut,bar)
                    for s,name in zip(streams,['LIGHT TOUCH','HEAVY TOUCH'])]
            comparison=np.concatenate(frames,axis=1)
            if process:process.stdin.write(comparison.tobytes())
            if i in (0,180,450,N-1):
                Image.fromarray(comparison).save(out/f'camera_1_light_vs_heavy_{i:04d}.jpg',quality=95)
            if i%120==0:print(args.object,'video',i,'/',N,round(time.monotonic()-start,1),flush=True)
        if process:
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
            pending.replace(video)
        if args.preview:return
        meta=dict(object=args.object,source_files=[str(v) for v in paths],camera='Cam1_274849',
            sides=['light_1','heavy_2'],frames=N,fps=FPS,duration_s=N/FPS,
            sensor_size=[W,H],video_size=[2*W,PANEL_H],
            normalization=f'One fixed pair-wide P1-P{args.upper_percentile:g} of every 640x512 sensor pixel from 8.0 to 15.0 s in both real recordings; exact uint16 histogram. Applied unchanged to the entire 15 s video, so the warmer hand can saturate. No per-frame normalization, calibrated temperatures, synthetic noise, or temporal smoothing.',
            color_map='inferno',global_limits_raw_counts=[lo,hi],histogram_pixels=total,
            normalization_window_s=[8.,15.],upper_percentile=args.upper_percentile,
            post_hand_quantiles_raw_counts=quantiles,
            source_frame_counts=[len(s.timestamps) for s in streams],
            max_timestamp_error_ms=[float(s.timing_error_ms.max()) for s in streams],
            selected_source_frame_indices=[s.selected.tolist() for s in streams],
            output=str(video),elapsed_s=time.monotonic()-start)
        (out/'camera_1_light_vs_heavy_global.json').write_text(json.dumps(meta,indent=2))
        print('COMPLETE',args.object,'limits',lo,hi,flush=True)


if __name__=='__main__':main()
