"""Export recorded camera-1 thermal and visible clips for a paired object capture."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from numpy.lib import format as npy_format
from PIL import Image, ImageDraw, ImageFont
import matplotlib

SOURCE_ROOT = Path('/data/sriram/thermal_tactile_force/force_interaction_data/source')
OUTPUT_ROOT = Path('/data/sriram/blender_renders')
FFMPEG = Path('/data/sriram/babybottle_thermal/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def encode(path: Path, size: tuple[int,int], fps: int):
    return subprocess.Popen([str(FFMPEG),'-y','-loglevel','error','-f','rawvideo',
        '-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-threads','4',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)


def reader(stream):
    version=npy_format.read_magic(stream)
    assert version==(1,0),version
    shape,order,dtype=npy_format.read_array_header_1_0(stream,max_header_size=30000)
    assert not order
    return shape,dtype,stream.tell()


def nearest(stamps, count, fps):
    target=stamps[0]+np.arange(count)/fps
    right=np.clip(np.searchsorted(stamps,target),0,len(stamps)-1)
    left=np.maximum(right-1,0)
    selected=np.where(abs(stamps[left]-target)<=abs(stamps[right]-target),left,right)
    assert np.all(np.diff(selected)>=0)
    return selected,np.abs(stamps[selected]-target)


def thermal(source: Path,out: Path,capture: str,trial: str,preview: bool):
    path=source/f'{capture}_Cam1_274849.npz'
    lut=(matplotlib.colormaps['inferno'](np.linspace(0,1,2048))[:,:3]*255).astype(np.uint8)
    count=3 if preview else 900
    with zipfile.ZipFile(path) as z,z.open('raw_thr_frames.npy') as stream:
        stamps=np.load(z.open('raw_thr_tstamps.npy'))
        shape,dtype,offset=reader(stream)
        assert shape[1:]==(514,640,1) and dtype==np.uint16
        assert len(stamps)==shape[0] and len(stamps)>=900
        selected,error=nearest(stamps,count,60)
        assert error.max()<.011,error.max()
        process=encode(out/'camera_1_real_normalized.mp4',(640,560),60)
        ranges=np.empty((count,2),np.float32)
        for i,source_i in enumerate(selected):
            stream.seek(offset+int(source_i)*514*640*2)
            data=stream.read(514*640*2)
            if len(data)!=514*640*2:raise EOFError((path,source_i))
            frame=np.frombuffer(data,np.uint16).reshape(514,640)[2:]
            lo,hi=(float(v) for v in np.percentile(frame,[1,99]))
            if hi<=lo:hi=lo+1.
            ranges[i]=lo,hi
            idx=np.rint(np.clip((frame.astype(np.float32)-lo)/(hi-lo),0,1)*2047).astype(np.uint16)
            image=Image.new('RGB',(640,560),(14,18,24));image.paste(Image.fromarray(lut[idx]),(0,40))
            draw=ImageDraw.Draw(image)
            draw.text((10,9),f'CAMERA 1 | REAL | {trial.upper()} | INFERNO P1-P99',
                font=ImageFont.truetype(FONT,17),fill='white')
            draw.text((10,530),f'{i/60:.2f} s | {lo:.0f} to {hi:.0f} raw counts',
                font=ImageFont.truetype(FONT,15),fill='#d4dce5')
            process.stdin.write(np.asarray(image).tobytes())
            if i in (0,180,450,count-1):image.save(out/f'camera_1_preview_{i:04d}.jpg',quality=95)
            if i%120==0:print(capture,'thermal',i,'/',count,flush=True)
        process.stdin.close();assert process.wait()==0
    np.save(out/'camera_1_normalization_ranges_counts.npy',ranges)
    return dict(source=str(path),source_frames=len(stamps),fps=60,frames=count,
        size=[640,560],sensor_size=[640,512],source_frame_indices=selected.tolist(),
        max_timing_error_ms=float(error.max()*1000),
        normalization='Per recorded camera-1 frame P1-P99 over raw 640x512 counts, then Inferno; telemetry rows removed.',
        file='camera_1_real_normalized.mp4')


def visible(source: Path,out: Path,capture: str,preview: bool):
    path=source/f'{capture}_vis.npz';count=3 if preview else 225
    with zipfile.ZipFile(path) as z,z.open('raw_vis_frames.npy') as stream:
        stamps=np.load(z.open('vis_cam_tstamps.npy'))
        ids=np.load(z.open('vis_frame_ids.npy'))
        pixel=str(np.load(z.open('vis_pixel_format.npy')).item())
        serial=str(np.load(z.open('vis_serial.npy')).item())
        shape,dtype,offset=reader(stream)
        assert shape[1:]==(3000,4000) and dtype==np.uint16
        assert len(stamps)==len(ids)==shape[0] and pixel=='BayerRG16' and serial=='24017931'
        selected,error=nearest(stamps,count,15)
        assert error.max()<.075,error.max()
        frame_bytes=3000*4000*2

        def decode(index):
            stream.seek(offset+int(index)*frame_bytes)
            data=stream.read(frame_bytes)
            if len(data)!=frame_bytes:raise EOFError((path,index))
            bayer=np.frombuffer(data,np.uint16).reshape(3000,4000)
            # For a top-left-red RGGB mosaic, this OpenCV code returns RGB array order.
            return cv2.cvtColor(bayer,cv2.COLOR_BayerRG2BGR)

        first=decode(0)
        board=cv2.resize(first,(1920,1440),interpolation=cv2.INTER_AREA)[500:1000,600:1250]
        whites=board[(board.min(2)>5000)&(board.max(2)<50000)]
        assert len(whites)>20000,len(whites)
        medians=np.median(whites,axis=0);gains=medians[1]/medians
        assert np.all((gains>.5)&(gains<3)),gains
        process=encode(out/'visible_real_normalized.mp4',(1920,1440),15)
        ranges=np.empty((count,2),np.float32);last=-1;image=None
        for i,source_i in enumerate(selected):
            if source_i!=last:
                rgb16=first if source_i==0 else decode(source_i)
                rgb=cv2.resize(rgb16,(1920,1440),interpolation=cv2.INTER_AREA).astype(np.float32)
                rgb*=gains
                luma=rgb[...,0]*.2126+rgb[...,1]*.7152+rgb[...,2]*.0722
                lo,hi=(float(v) for v in np.percentile(luma[::4,::4],[1,99]))
                if hi<=lo:hi=lo+1.
                balanced=np.clip((rgb-lo)/(hi-lo),0,1)
                image=np.rint(balanced**.65*255).astype(np.uint8)
                last=source_i
            ranges[i]=lo,hi
            process.stdin.write(image.tobytes())
            if i in (0,45,90,135,count-1):Image.fromarray(image).save(out/f'visible_preview_{i:04d}.jpg',quality=95)
            if i%45==0:print(capture,'visible',i,'/',count,flush=True)
        process.stdin.close();assert process.wait()==0
    np.save(out/'visible_normalization_ranges_counts.npy',ranges)
    return dict(source=str(path),source_frames=len(stamps),fps=15,frames=count,
        size=[1920,1440],source_frame_indices=selected.tolist(),
        max_timing_error_ms=float(error.max()*1000),camera_serial=serial,
        pixel_format=pixel,missing_frame_ids=int(ids[-1]-ids[0]+1-len(ids)),
        white_balance_raw_medians=medians.tolist(),rgb_gains=gains.tolist(),
        normalization='Per output frame P1-P99 of white-balanced luminance, sampled every fourth pixel, followed by fixed gamma 0.65; repeated source frames retain the same image.',
        file='visible_real_normalized.mp4')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--capture',required=True,help='For example a_tennis_ball_1')
    p.add_argument('--mode',required=True,choices=('thermal','visible'))
    p.add_argument('--preview',action='store_true')
    p.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    p.add_argument('--output-root',type=Path,default=OUTPUT_ROOT)
    a=p.parse_args()
    base,suffix=a.capture.rsplit('_',1)
    assert suffix in ('1','2')
    trial='light' if suffix=='1' else 'heavy'
    source=a.source_root/a.capture
    assert source.is_dir()
    out=a.output_root/base/f'{trial}_multiview'
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    detail=thermal(source,out,a.capture,trial,a.preview) if a.mode=='thermal' else visible(source,out,a.capture,a.preview)
    detail.update(capture=a.capture,object=base,trial=trial,elapsed_s=time.monotonic()-start)
    (out/f'{a.mode}_real_normalized.json').write_text(json.dumps(detail,indent=2))
    if not a.preview:
        readme=out/'README.md';existing=readme.read_text() if readme.exists() else f'# {base}: {trial} recorded capture\n'
        line=(f'\n- [{a.mode.title()} recording]({detail["file"]}): '
              f'{detail["frames"]/detail["fps"]:.0f} s at {detail["fps"]} fps. '
              'Independently normalized per frame.\n')
        if f']({detail["file"]})' not in existing:readme.write_text(existing+line)
    print('COMPLETE',a.capture,a.mode,flush=True)


if __name__=='__main__':main()
