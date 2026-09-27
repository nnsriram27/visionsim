"""Blender: animated close-range Livox-inspired scan over the original RGB camera.

Adapted from Srirams-MBP-6:scripts/render_livox_avia_fisheye.py (e6758d18).
No source blend is saved. All deliverables stay flat in the existing trial folder.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import bpy
import numpy as np
from embreex import mesh_construction, rtcore_scene
from PIL import Image, ImageDraw, ImageFont
import matplotlib
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

MODEL=dict(name='Close-range Livox Avia-inspired geometric approximation',
    horizontal_fov_deg=70.4,vertical_fov_deg=77.2,point_rate_hz=240000,
    range_precision_sigma_m=.02,minimum_range_m=.1,maximum_range_m=450.,
    returns=1,scan_pattern='Temporally continuous R2 low-discrepancy approximation')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FFMPEG='/data/sriram/babybottle_thermal/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
TRIAL=os.environ.get('BABYBOTTLE_TRIAL','heavy')
assert TRIAL in ('light','heavy')

def angular_pattern(start,count):
    index=start+np.arange(count,dtype=np.float64)
    u=np.mod(.5+index*.7548776662466927,1.);v=np.mod(.5+index*.5698402909980532,1.)
    directions=np.column_stack((np.tan(np.deg2rad((u-.5)*70.4)),np.tan(np.deg2rad((v-.5)*77.2)),-np.ones(count)))
    return directions/np.linalg.norm(directions,axis=1,keepdims=True)

def mesh_scene(objects,graph):
    vertices=[];triangles=[];offset=0;names=[]
    for obj in objects:
        if obj.hide_render or not obj.visible_get():continue
        evaluated=obj.evaluated_get(graph)
        mesh=evaluated.to_mesh()
        if mesh is None:continue
        if not mesh.vertices or not mesh.polygons:
            evaluated.to_mesh_clear();continue
        coords=np.empty(len(mesh.vertices)*3,dtype=np.float64);mesh.vertices.foreach_get('co',coords)
        M=np.asarray(evaluated.matrix_world);coords=coords.reshape(-1,3)@M[:3,:3].T+M[:3,3]
        mesh.calc_loop_triangles();indices=np.empty(len(mesh.loop_triangles)*3,dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices',indices)
        vertices.append(coords);triangles.append(indices.reshape(-1,3)+offset);offset+=len(coords)
        names.append(obj.name);evaluated.to_mesh_clear()
    if not vertices:return None,None,dict(objects=names,vertices=0,triangles=0)
    vv=np.ascontiguousarray(np.vstack(vertices),dtype=np.float32);ff=np.ascontiguousarray(np.vstack(triangles),dtype=np.uint32)
    scene=rtcore_scene.EmbreeScene();owner=mesh_construction.TriangleMesh(scene,vv,ff)
    return scene,owner,dict(objects=names,vertices=len(vv),triangles=len(ff))

def distance(scene,origins,directions):
    if scene is None:return np.full(len(directions),np.inf)
    result=np.asarray(scene.run(origins,directions,query='DISTANCE')).reshape(-1).astype(float)
    result[(result<=0)|~np.isfinite(result)]=np.inf
    return result

def project_sensor(points,K):
    z=-points[:,2]
    return np.column_stack((K[0,0]*points[:,0]/z+K[0,2],K[1,2]-K[1,1]*points[:,1]/z))

def writer(path,w,h):
    return subprocess.Popen([FFMPEG,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{w}x{h}','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)

def decorate(rgb,title,t,count=None):
    h,w=rgb.shape[:2];im=Image.new('RGB',(w,h+94),(14,18,24));im.paste(Image.fromarray(rgb),(0,48))
    draw=ImageDraw.Draw(im);font=ImageFont.truetype(FONT,21);small=ImageFont.truetype(FONT,16)
    draw.text((14,12),title,font=font,fill='white');draw.text((w-120,15),f'{t:.2f} s',font=small,fill='white')
    text=f'Original visible camera | {TRIAL} press' if count is None else f'{count} projected returns | range 0.25 - 1.25 m | 33.3 ms scan'
    draw.text((14,h+56),text,font=small,fill='#c4cdd7')
    if count is not None:
        bar=(matplotlib.colormaps['turbo'](np.linspace(0,1,180))[:,:3]*255).astype('uint8')
        im.paste(Image.fromarray(np.repeat(bar[None],12,axis=0)),(w-195,h+62))
    return np.asarray(im)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--scan-only',action='store_true',help='Generate ranges for third-person videos without RGB/overlay rendering')
    parser.add_argument('--output',type=Path,default=Path(f'/data/sriram/babybottle_thermal/{TRIAL}_multiview'))
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    out=args.output;assert out.is_dir();source=Path(bpy.data.filepath);source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    assert not (args.preview and args.scan_only)
    scene=bpy.data.scenes['01_Light_Force' if TRIAL=='light' else '02_Heavy_Force'];bpy.context.window.scene=scene
    camera=scene.objects['visible_24017931'];scene.camera=camera
    scene.render.resolution_x=1000;scene.render.resolution_y=750;scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=1;scene.render.pixel_aspect_y=1;scene.render.use_border=False
    scene.render.engine='CYCLES';scene.cycles.samples=args.samples;scene.cycles.use_denoising=True
    scene.render.use_persistent_data=True
    if not args.scan_only:
        prefs=bpy.context.preferences.addons['cycles'].preferences;prefs.compute_device_type='OPTIX';prefs.get_devices()
        devices=[d for d in prefs.devices if d.type=='OPTIX'];assert devices,'No OptiX GPU available'
        for d in prefs.devices:d.use=d==devices[0]
        scene.cycles.device='GPU';scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.frame_set(1);bpy.context.view_layer.update()
    K=np.array(json.loads(camera['K_json']),dtype=float);K[:2]*=.25
    pose=np.array(camera.matrix_world);rotation=pose[:3,:3];origin=pose[:3,3];unit=scene.unit_settings.scale_length
    geometry=[o for o in scene.objects if o.type in {'MESH','CURVE','FONT','SURFACE'}]
    dynamic=[o for o in geometry if o.animation_data or (o.type=='MESH' and o.data.shape_keys)]
    static=[o for o in geometry if o not in dynamic]
    assert len(dynamic)==1 and TRIAL in dynamic[0].name,(str([o.name for o in dynamic]))
    static_bvh,static_owner,static_info=mesh_scene(static,bpy.context.evaluated_depsgraph_get());assert static_owner is not None
    # Check the custom K against Blender's actual projection before producing overlays.
    test=angular_pattern(100,20)*.7
    expected=project_sensor(test,K);blender_uv=[]
    for point in origin+test@rotation.T:
        co=world_to_camera_view(scene,camera,Vector(point));blender_uv.append([co.x*1000,(1-co.y)*750])
    projection_error=float(np.max(np.abs(expected-np.asarray(blender_uv))));assert projection_error<.002,projection_error
    # Analytical plane check exercises Embree distance units and first-surface rays.
    test_scene=rtcore_scene.EmbreeScene();test_owner=mesh_construction.TriangleMesh(test_scene,
        np.array([[-2,-2,-1],[2,-2,-1],[2,2,-1],[-2,2,-1]],np.float32),np.array([[0,1,2],[0,2,3]],np.uint32))
    rays=np.ascontiguousarray(angular_pattern(0,100),np.float32)
    plane_distance=distance(test_scene,np.zeros_like(rays),rays)
    assert np.max(np.abs(plane_distance-1/(-rays[:,2])))<1e-5
    indices=[144] if args.preview else range(450);writers={}
    if not args.preview and not args.scan_only:
        writers={name:writer(out/f'lidar_visible_{name}.mp4',2000 if name=='comparison' else 1000,844)
                 for name in ['rgb','overlay','comparison']}
    shape=(len(indices),8000);ranges=np.full(shape,np.nan,np.float32);ideal=ranges.copy()
    uv_all=np.full(shape+(2,),np.nan,np.float32);object_id=np.full(shape,-1,np.int8)
    reports=[];start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='babybottle_lidar_') as tempdir:
        for row,i in enumerate(indices):
            t=i/30;scene.frame_set(1+2*i);bpy.context.view_layer.update()
            assert np.max(np.abs(np.array(camera.matrix_world)-pose))<1e-7,'Camera moved'
            graph=bpy.context.evaluated_depsgraph_get();hand_bvh,hand_owner,hand_info=mesh_scene(dynamic,graph)
            directions=angular_pattern(i*8000,8000);world=np.ascontiguousarray(directions@rotation.T,np.float32)
            origins=np.ascontiguousarray(np.broadcast_to(origin,world.shape),np.float32)
            ds=distance(static_bvh,origins,world);dh=distance(hand_bvh,origins,world)
            d=np.minimum(ds,dh)*unit;valid=np.isfinite(d)&(d>=.1)&(d<=450)
            rng=np.random.default_rng(np.random.SeedSequence([20260924,i]))
            noisy=np.clip(d[valid]+rng.normal(0,.02,valid.sum()),.1,450)
            points=directions[valid]*noisy[:,None]/unit;uv=project_sensor(points,K)
            assert np.max(abs(uv-project_sensor(directions[valid],K)))<1e-8
            ranges[row,valid]=noisy;ideal[row,valid]=d[valid];uv_all[row,valid]=uv
            object_id[row,valid]=np.where(dh[valid]<ds[valid],1,0)
            xy=np.floor(uv).astype(int);inside=(xy[:,0]>=1)&(xy[:,0]<999)&(xy[:,1]>=1)&(xy[:,1]<749)
            reports.append(dict(frame=i,time_s=t,blender_frame=1+2*i,returns=int(valid.sum()),projected=int(inside.sum()),
                hand_returns=int((object_id[row]==1).sum()),hand_visible=hand_owner is not None,
                range_min_median_max_m=[float(noisy.min()),float(np.median(noisy)),float(noisy.max())]))
            if args.scan_only:
                if i%60==0:print('LIDAR SCAN',TRIAL,i,flush=True)
                del hand_bvh,hand_owner
                continue
            # Range noise moves a co-located return along its ray, not sideways in RGB.
            pix=xy[inside];colors=(matplotlib.colormaps['turbo'](np.clip((noisy[inside]-.25)/1.,0,1))[:,:3]*255).astype('uint8')
            order=np.argsort(noisy[inside])[::-1];pix=pix[order];colors=colors[order]
            png=Path(tempdir)/'rgb.png';scene.render.filepath=str(png);bpy.ops.render.render(write_still=True)
            rgb=np.array(Image.open(png).convert('RGB'));overlay=rgb.copy()
            for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:overlay[pix[:,1]+dy,pix[:,0]+dx]=colors
            a=decorate(rgb,'VISIBLE RGB | original Blender scene',t)
            b=decorate(overlay,'LiDAR on RGB | close-range Avia-inspired',t,int(inside.sum()))
            comparison=np.concatenate([a,b],axis=1)
            if args.preview:
                Image.fromarray(comparison).save(out/'lidar_visible_preview.jpg',quality=95)
            else:
                for name,frame in [('rgb',a),('overlay',b),('comparison',comparison)]:writers[name].stdin.write(frame.tobytes())
                if i in [0,90,144,240]:Image.fromarray(comparison).save(out/f'lidar_visible_{i:04d}.jpg',quality=95)
            print('LIDAR_FRAME',i,'projected',int(inside.sum()),'elapsed',round(time.monotonic()-start,1),flush=True)
            del hand_bvh,hand_owner
    for proc in writers.values():
        proc.stdin.close();assert proc.wait()==0,'Video encoding failed'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    if args.preview:return
    np.savez_compressed(out/'lidar_visible_returns.npz',range_m=ranges,ideal_range_m=ideal,uv_pixels=uv_all,
        object_id=object_id,frame_time_s=np.arange(450)/30,firing_offset_s=np.arange(8000)/240000,
        K=K,sensor_to_world=pose)
    metadata=dict(source_blend=str(source),source_sha256=source_hash,scene=scene.name,camera=camera.name,trial=TRIAL,scan_only=args.scan_only,
        inherited_model_source='VisionSim Srirams-MBP-6 e6758d18271491c354ba9bd009663a894e953ec5 scripts/render_livox_avia_fisheye.py',
        model=MODEL,frames=450,fps=30,width=1000,height=750,unit_scale_m=unit,static_geometry=static_info,
        dynamic_objects=[o.name for o in dynamic],projection_error_px=projection_error,seed=20260924,
        cycles_samples=None if args.scan_only else args.samples,render_device=None if args.scan_only else devices[0].name,scan_integration_s=1/30,
        coordinates='Sensor +X right, +Y up, -Z forward; pixel origin top left. Exactly co-located with RGB camera.',
        data='Rows=video frames, columns=ray firing index; NaN means no return. object_id -1 absent, 0 static, 1 hand. XYZ can be reconstructed from UV, K and range. Range is radial, not optical-axis depth.',
        limitations=['Same near-field 0.1 m cutoff as room approximation, not a validated physical Avia at this working distance.',
            '2 cm noise is inherited from room model; not characterized for this bottle at sub-meter range.',
            'Non-repetitive sampling approximated by R2, not exact Livox rosette optics.',
            'Every visible surface is opaque to single-return ray casting; glass transmission, material-dependent missed returns, multipath, intensity, divergence and atmosphere are not modeled.',
            'Animated hand is rebuilt each 30 fps frame; firing times are stored but geometry is frozen within the 33.3 ms scan. No cross-frame accumulation or synthetic motion trails.',
            'Points outside the RGB field of view are retained in numeric data but omitted from the overlay. Five-pixel cross splats are visualization only.'],
        references=['https://www.livoxtech.com/avia/specs','https://www.livoxtech.com/avia/downloads'],frames_report=reports,
        elapsed_s=time.monotonic()-start)
    (out/'lidar_visible.json').write_text(json.dumps(metadata,indent=2));print('LIDAR COMPLETE',flush=True)

if __name__=='__main__':main()
