"""Short rectified stereo clips, following the room's 65-mm geometric stereo setup."""
import hashlib,json,os,sys,tempfile,time
from pathlib import Path
import bpy
import numpy as np
import matplotlib
from PIL import Image,ImageDraw,ImageFont
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
sys.path.insert(0,str(Path(__file__).parent))
from render_lidar_visible import mesh_scene,distance,writer,FONT

TRIAL=os.environ.get('BABYBOTTLE_TRIAL','heavy');assert TRIAL in ('light','heavy')
OUT=Path(f'/data/sriram/babybottle_thermal/{TRIAL}_multiview')
W,H=800,600
BASELINE=.065
LIMITS=(20.,100.)

def panel(rgb,title,t,disparity=False):
    im=Image.new('RGB',(W,680),(14,18,24));im.paste(Image.fromarray(rgb),(0,42));d=ImageDraw.Draw(im)
    d.text((12,10),f'{TRIAL.upper()} | {title}',font=ImageFont.truetype(FONT,20),fill='white')
    d.text((660,13),f'{t:.2f} s',font=ImageFont.truetype(FONT,16),fill='white')
    label='Parallel stereo | 65 mm baseline | original RGB rig center'
    if disparity:
        label='Ideal left disparity | 20 px (far) to 100 px (near)'
        bar=(matplotlib.colormaps['turbo'](np.linspace(0,1,150))[:,:3]*255).astype('uint8')
        im.paste(Image.fromarray(np.repeat(bar[None],14,axis=0)),(638,653))
    d.text((12,650),label,font=ImageFont.truetype(FONT,16),fill='#b9cfde')
    return np.asarray(im)

def main():
    source=Path(bpy.data.filepath);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    scene=bpy.data.scenes['01_Light_Force' if TRIAL=='light' else '02_Heavy_Force'];bpy.context.window.scene=scene
    original=scene.objects['visible_24017931'];center=np.array(original.matrix_world);unit=scene.unit_settings.scale_length
    assert original.data.type=='PERSP'
    eyes=[];poses=[]
    for name,offset in [('left',-.5),('right',.5)]:
        eye=original.copy();eye.data=original.data.copy();eye.animation_data_clear();eye.constraints.clear()
        scene.collection.objects.link(eye);eye.name=f'stereo_{name}_temporary'
        pose=center.copy();pose[:3,3]+=center[:3,0]*offset*BASELINE/unit
        eye.matrix_world=Matrix(pose);eyes.append(eye);poses.append(pose)
    assert np.allclose(poses[0][:3,:3],poses[1][:3,:3])
    assert abs(np.linalg.norm(poses[1][:3,3]-poses[0][:3,3])*unit-BASELINE)<1e-7
    scene.render.resolution_x=W;scene.render.resolution_y=H;scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1;scene.render.use_border=False
    scene.render.engine='CYCLES';scene.cycles.samples=64;scene.cycles.use_denoising=True
    scene.cycles.seed=0;scene.cycles.use_animated_seed=False;scene.render.use_persistent_data=True
    scene.render.use_motion_blur=False
    prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
    devices=[d for d in prefs.devices if d.type=='OPTIX'];assert devices
    for d in prefs.devices:d.use=d==devices[0]
    scene.cycles.device='GPU';scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.image_settings.color_depth='8'
    K=np.array(json.loads(original['K_json']),float);K[:2]*=W/4000
    yy,xx=np.mgrid[:H,:W];rays=np.stack([(xx+.5-K[0,2])/K[0,0],-(yy+.5-K[1,2])/K[1,1],-np.ones((H,W))],-1).reshape(-1,3)
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    world=np.ascontiguousarray(rays@center[:3,:3].T,np.float32)
    origins=np.ascontiguousarray(np.broadcast_to(poses[0][:3,3],world.shape),np.float32)
    # Independently verify K and horizontal epipolar/disparity geometry using Blender.
    errors=[]
    for point in rays[::5000]*.8:
        p=poses[0][:3,3]+center[:3,:3]@point
        a=world_to_camera_view(scene,eyes[0],Vector(p));b=world_to_camera_view(scene,eyes[1],Vector(p))
        errors.extend([abs((a.x-b.x)*W-K[0,0]*BASELINE/(-point[2]*unit)),abs(a.y-b.y)*H,
            abs(a.x*W-(K[0,0]*point[0]/-point[2]+K[0,2]))])
    assert max(errors)<.002,max(errors)
    geometry=[o for o in scene.objects if o.type in {'MESH','CURVE','FONT','SURFACE'}]
    dynamic=[o for o in geometry if o.animation_data or (o.type=='MESH' and o.data.shape_keys)]
    static=[o for o in geometry if o not in dynamic];assert len(dynamic)==1
    scene.frame_set(1);bpy.context.view_layer.update()
    static_bvh,static_owner,_=mesh_scene(static,bpy.context.evaluated_depsgraph_get())
    ds=distance(static_bvh,origins,world)
    start,end=(1.,5.5) if TRIAL=='light' else (1.5,7.5);count=round((end-start)*30)
    encoders={n:writer(OUT/f'{n}.mp4',w,680) for n,w in [('stereo_sbs',1600),('stereo_disparity_turbo',800),('stereo_rgb_disparity',1600)]}
    lut=(matplotlib.colormaps['turbo'](np.linspace(0,1,2048))[:,:3]*255).astype('uint8')
    disparities=[];records=[];clock=time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f'babybottle_stereo_{TRIAL}_') as temp:
        png=Path(temp)/'eye.png'
        for i in range(count):
            t=start+(i+1)/30;scene.frame_set(round(1+t*60));bpy.context.view_layer.update()
            assert np.max(abs(np.array(original.matrix_world)-center))<1e-7
            hand_bvh,hand_owner,_=mesh_scene(dynamic,bpy.context.evaluated_depsgraph_get())
            ranges=np.minimum(ds,distance(hand_bvh,origins,world));depth=ranges*(-rays[:,2])*unit
            valid=np.isfinite(depth)&(depth>0)&(ranges<1e10)
            disp=np.full(len(depth),np.nan,np.float32);disp[valid]=K[0,0]*BASELINE/depth[valid]
            disp=disp.reshape(H,W);disparities.append(disp.astype(np.float16))
            norm=np.nan_to_num(np.clip((disp-LIMITS[0])/(LIMITS[1]-LIMITS[0]),0,1))
            heat=lut[np.rint(norm*2047).astype(int)];heat[~valid.reshape(H,W)]=0
            rgb=[]
            for eye in eyes:
                scene.camera=eye;scene.render.filepath=str(png);bpy.ops.render.render(write_still=True)
                rgb.append(np.array(Image.open(png).convert('RGB')))
            a=panel(rgb[0],'Left RGB',t);b=panel(rgb[1],'Right RGB',t);c=panel(heat,'Disparity / Turbo',t,True)
            frames={'stereo_sbs':np.concatenate([a,b],1),'stereo_disparity_turbo':c,'stereo_rgb_disparity':np.concatenate([a,c],1)}
            for n,frame in frames.items():encoders[n].stdin.write(frame.tobytes())
            if i in [0,45,75,count-1]:Image.fromarray(frames['stereo_rgb_disparity']).save(OUT/f'stereo_preview_{i:04d}.jpg',quality=95)
            records.append(dict(frame=i,time_s=t,disparity_percentiles_px=np.nanpercentile(disp,[1,50,99]).tolist(),
                eye_mean_absolute_difference=float(np.abs(rgb[0].astype(float)-rgb[1]).mean()),hand_visible=not dynamic[0].hide_render))
            if i%15==0:print('STEREO',TRIAL,i,'/',count,'elapsed',round(time.monotonic()-clock,1),flush=True)
            del hand_bvh,hand_owner
    for proc in encoders.values():proc.stdin.close();assert proc.wait()==0
    np.savez_compressed(OUT/'stereo_disparity.npz',disparity_px=np.stack(disparities),K=K,left_to_world=poses[0],right_to_world=poses[1],
        baseline_m=BASELINE,frame_time_s=start+(np.arange(count)+1)/30)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
    meta=dict(trial=TRIAL,source_blend=str(source),source_sha256=sha,camera=original.name,scene=scene.name,
        baseline_m=BASELINE,center_to_world=center.tolist(),left_to_world=poses[0].tolist(),right_to_world=poses[1].tolist(),K=K.tolist(),
        projection='Parallel rectified perspective stereo, same original RGB intrinsics',unit_scale_m=unit,
        fps=30,frames=count,duration_s=end-start,scene_interval_s=[start,end],eye_resolution=[W,H],
        disparity='Ideal left-reference horizontal disparity d=u_left-u_right=fx*baseline_m/optical_axis_depth_m',
        color_map='turbo',fixed_color_limits_px=LIMITS,projection_check_max_error_px=max(errors),records=records,
        limitations=['Geometric first-surface ground truth, not stereo matching or calibrated hardware.',
            'Glass/reflections are rendered in RGB but disparity follows mesh surfaces, not refracted/reflected apparent depth.',
            'Disparity includes left-visible points occluded or outside the right image; no correspondence-validity mask.',
            'Numeric disparity is float16 pixel units; invalid rays are NaN. Colors clipped to fixed display limits.',
            'Original hand poses and visibility switches preserved.'],cycles_samples=64,elapsed_s=time.monotonic()-clock)
    (OUT/'stereo.json').write_text(json.dumps(meta,indent=2));print('STEREO COMPLETE',TRIAL,flush=True)

if __name__=='__main__':main()
