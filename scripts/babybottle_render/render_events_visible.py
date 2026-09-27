"""Blender + native VisionSim DVS: short RGB-camera light/heavy event sequences."""
import hashlib,json,logging,os,subprocess,sys,tempfile,time
from pathlib import Path
import bpy
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from visionsim.emulate.dvs import EventEmulator

TRIAL=os.environ.get('BABYBOTTLE_TRIAL','heavy');assert TRIAL in ('light','heavy')
OUT=Path(f'/data/sriram/babybottle_thermal/{TRIAL}_multiview')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FFMPEG='/data/sriram/babybottle_thermal/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
PARAMS=dict(pos_thres=.2,neg_thres=.2,sigma_thres=.02,cutoff_hz=0.,leak_rate_hz=0.,
    shot_noise_rate_hz=0.,refractory_period_s=0.,photoreceptor_noise=False,
    leak_jitter_fraction=0.,noise_rate_cov_decades=0.,seed=20260914)
W,H=800,600

def encoder(path,width):
    return subprocess.Popen([FFMPEG,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x680','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','16',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

def panel(rgb,title,t,event=False):
    im=Image.new('RGB',(W,680),(14,18,24));im.paste(Image.fromarray(rgb),(0,42));draw=ImageDraw.Draw(im)
    draw.text((12,10),f'{TRIAL.upper()} | {title}',font=ImageFont.truetype(FONT,20),fill='white')
    draw.text((638,12),f'Scene {t:.2f} s',font=ImageFont.truetype(FONT,15),fill='white')
    label='Blue: ON (+) | Red: OFF (-) | 33.3 ms window' if event else 'Fixed visible_24017931 viewpoint | original scene'
    draw.text((12,651),label,font=ImageFont.truetype(FONT,17),fill='#b9cfde')
    return np.asarray(im)

def main():
    logging.getLogger('visionsim.emulate.dvs.v2e.emulator').setLevel(logging.ERROR)
    source=Path(bpy.data.filepath);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    scene=bpy.data.scenes['01_Light_Force' if TRIAL=='light' else '02_Heavy_Force'];bpy.context.window.scene=scene
    camera=scene.objects['visible_24017931'];scene.camera=camera;pose=np.array(camera.matrix_world)
    hand=next(o for o in scene.objects if o.type=='MESH' and o.data.shape_keys)
    scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=1;scene.render.use_border=False
    scene.render.engine='CYCLES';scene.cycles.samples=64;scene.cycles.use_denoising=True
    scene.cycles.seed=0;scene.cycles.use_animated_seed=False;scene.render.use_persistent_data=True
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    devices=[d for d in prefs.devices if d.type=='OPTIX'];assert devices
    for d in prefs.devices:d.use=d==devices[0]
    scene.cycles.device='GPU';scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.image_settings.color_depth='8'
    start,end=(1.,5.5) if TRIAL=='light' else (1.5,7.5)
    output_count=round((end-start)*30);input_count=output_count*4+1
    event_encoder=encoder(OUT/'event_camera.mp4',W);compare_encoder=encoder(OUT/'event_rgb_comparison.mp4',2*W)
    emulator=EventEmulator(**PARAMS);viz=np.full((H,W,3),255,np.uint8)
    chunks=[];records=[];window_on=window_off=0;start_clock=time.monotonic()
    # Static-scene control independently checks emulator behavior, without render noise.
    control=EventEmulator(**PARAMS);flat=np.full((16,16),100.,np.float32)
    assert control.generate_events(flat,0.) is None
    assert len(control.generate_events(flat,1/120))==0
    with tempfile.TemporaryDirectory(prefix=f'babybottle_events_{TRIAL}_') as temp:
        png=Path(temp)/'frame.png'
        for j in range(input_count):
            t=start+j/120;blender_frame=1+t*60
            base=int(np.floor(blender_frame));scene.frame_set(base,subframe=blender_frame-base)
            bpy.context.view_layer.update();assert np.max(abs(np.array(camera.matrix_world)-pose))<1e-7
            scene.render.filepath=str(png);bpy.ops.render.render(write_still=True)
            rgb=np.array(Image.open(png).convert('RGB'))
            luma=(rgb.astype(np.float64)*np.array([.2126,.7152,.0722])).sum(axis=2)
            events=emulator.generate_events(luma,j/120)
            if events is not None and len(events):
                assert np.isfinite(events).all()
                chunks.append(events.copy());x=events[:,1].astype(int);y=events[:,2].astype(int);pos=events[:,3]>0
                assert x.min()>=0 and x.max()<W and y.min()>=0 and y.max()<H
                viz[y[~pos],x[~pos]]=[255,0,0];viz[y[pos],x[pos]]=[0,0,255]
                window_on+=int(pos.sum());window_off+=int((~pos).sum())
            if j and j%4==0:
                a=panel(rgb,'RGB reference',t);b=panel(viz,'VisionSim events',t,True)
                event_encoder.stdin.write(b.tobytes());compare=np.concatenate([a,b],axis=1);compare_encoder.stdin.write(compare.tobytes())
                index=j//4-1
                records.append(dict(frame=index,scene_time_s=t,on=window_on,off=window_off,
                    visible_on_pixels=int(np.all(viz==[0,0,255],axis=-1).sum()),
                    visible_off_pixels=int(np.all(viz==[255,0,0],axis=-1).sum()),hand_visible=not hand.hide_render))
                if index in [0,45,75,output_count-1]:Image.fromarray(compare).save(OUT/f'event_preview_{index:04d}.jpg',quality=95)
                viz.fill(255);window_on=window_off=0
                if index%15==0:print('EVENT_VIDEO',TRIAL,index,'/',output_count,'events',sum(len(c) for c in chunks),'elapsed',round(time.monotonic()-start_clock,1),flush=True)
    for proc in [event_encoder,compare_encoder]:proc.stdin.close();assert proc.wait()==0
    events=np.concatenate(chunks) if chunks else np.empty((0,4),np.float32)
    assert len(records)==output_count
    np.savez_compressed(OUT/'event_camera_events.npz',t_scene_s=events[:,0].astype(np.float64)+start,
        x=events[:,1].astype(np.uint16),y=events[:,2].astype(np.uint16),polarity=events[:,3].astype(np.int8))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    model_file=Path(sys.modules[EventEmulator.__module__].__file__)
    meta=dict(trial=TRIAL,scene=scene.name,camera=camera.name,source_blend=str(source),source_sha256=source_hash,
        scene_interval_s=[start,end],duration_s=end-start,output_frames=output_count,fps=30,input_fps=120,
        input_frames=input_count,sensor_size=[W,H],event_video_size=[W,680],comparison_video_size=[2*W,680],
        model='Native VisionSim EventEmulator, v2e-derived; clean preset from room native-white event rendering',
        model_file=str(model_file),model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(),parameters=PARAMS,
        temporal_source='Direct Blender 120 Hz subframe rendering; no compressed-video input, RIFE, or optical-flow interpolation.',
        rendering=dict(engine='Cycles',samples=64,denoising=True,seed=0,animated_seed=False,view_transform=scene.view_settings.view_transform,
            exposure=scene.view_settings.exposure,gamma=scene.view_settings.gamma),
        visualization='Native white background; ON blue, OFF red. Binary occupancy over four 120 Hz steps, not event count brightness. Headers excluded from emulation.',
        window_timing='Frame i accumulates events in (start+i/30, start+(i+1)/30]; RGB shown at interval endpoint. Event timestamps use original scene seconds.',
        total_on=int(np.sum(events[:,3]>0)),total_off=int(np.sum(events[:,3]<0)),records=records,
        limitations=['RGB-derived behavioral emulation, not hardware capture or a calibrated event camera.',
            '120 Hz source sampling cannot recover faster unseen brightness changes; interpolated event timestamps are model estimates.',
            'Same display-RGB luminance convention as room preview; not calibrated linear sensor irradiance.',
            'Clean preset has threshold mismatch but no leak/shot noise. Cycles sampling and denoising can still introduce rendering artifacts.',
            'Original hand visibility switches can create abrupt entry/exit events. No pose or visibility edits were made.',
            'Repeated events at a pixel are retained in NPZ but collapsed in the video preview.'],
        elapsed_s=time.monotonic()-start_clock)
    (OUT/'event_camera.json').write_text(json.dumps(meta,indent=2));print('EVENT CAMERA COMPLETE',TRIAL,flush=True)

if __name__=='__main__':main()
