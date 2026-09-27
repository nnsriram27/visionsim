"""Third-person point-cloud presentation of the existing heavy LiDAR returns.

Room-style static scan history, current moving-hand returns only, no RGB backdrop.
Produces measured/noisy and explicitly ideal-range companion views.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import imageio_ffmpeg

TRIAL=os.environ.get('BABYBOTTLE_TRIAL','heavy')
assert TRIAL in ('light','heavy')
OUT=Path(f'/data/sriram/babybottle_thermal/{TRIAL}_multiview')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def normalize(v):return v/np.linalg.norm(v)

class View:
    def __init__(self,eye,target,fov=45):
        self.eye=np.array(eye);self.forward=normalize(np.array(target)-self.eye)
        self.right=normalize(np.cross(self.forward,[0,0,1]));self.up=np.cross(self.right,self.forward)
        self.focal=640/np.tan(np.deg2rad(fov/2))
    def project(self,xyz):
        relative=xyz-self.eye;z=relative@self.forward
        uv=np.column_stack((640+self.focal*(relative@self.right)/np.maximum(z,1e-9),360-self.focal*(relative@self.up)/np.maximum(z,1e-9)))
        return uv,z

def cloud(view,xyz,colors,radius,pixels,zbuffer):
    if not len(xyz):return
    uv,z=view.project(xyz);xy=np.rint(uv).astype(int)
    good=(z>0)&(xy[:,0]>=radius)&(xy[:,0]<1280-radius)&(xy[:,1]>=radius)&(xy[:,1]<720-radius)
    xy,z,colors=xy[good],z[good],colors[good]
    # Keep the nearest point for each splat pixel, across both history/current.
    for dx,dy in [(x,y) for x in range(-radius,radius+1) for y in range(-radius,radius+1) if x*x+y*y<=radius*radius]:
        ix=(xy[:,1]+dy)*1280+xy[:,0]+dx
        order=np.argsort(z,kind='stable');_,first=np.unique(ix[order],return_index=True);selected=order[first]
        ix=ix[selected];depth=z[selected];front=depth<zbuffer[ix]
        pixels[ix[front]]=colors[selected[front]];zbuffer[ix[front]]=depth[front]

def line(draw,view,a,b,color,width=1):
    uv,z=view.project(np.array([a,b]))
    if np.all(z>0):draw.line([tuple(uv[0]),tuple(uv[1])],fill=color,width=width)

def background(view,close=False):
    im=Image.new('RGB',(1280,720),(10,16,23));draw=ImageDraw.Draw(im)
    step=.025 if close else .1
    for value in np.arange(-.5,.801,step):
        line(draw,view,[value,-.6,-.015],[value,.6,-.015],(24,35,45))
        line(draw,view,[-.5,value,-.015],[.8,value,-.015],(24,35,45))
    return np.asarray(im).copy()

def range_colors(ranges):
    """Radial distance from the scanner, independent of display view or height."""
    return (matplotlib.colormaps['turbo'](np.clip((np.asarray(ranges)-.25)/1.,0,1))[:,:3]*255).astype('uint8')

def render(view,history,current,ranges,close=False):
    pixels=background(view,close).reshape(-1,3);zbuffer=np.full(1280*720,np.inf)
    history_colors=np.tile(np.array([76,112,136],np.uint8),(len(history),1))
    cloud(view,history,history_colors,2 if close else 1,pixels,zbuffer)
    color=range_colors(ranges)
    cloud(view,current,color,4 if close else 2,pixels,zbuffer)
    return Image.fromarray(pixels.reshape(720,1280,3))

def writer(path):
    w=imageio_ffmpeg.write_frames(str(path),(1920,800),fps=30,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
        macro_block_size=1,output_params=['-crf','18','-preset','fast','-threads','2','-movflags','+faststart'])
    w.send(None);return w

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    source=OUT/'lidar_visible_returns.npz';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    data=np.load(source);pose=data['sensor_to_world'];K=data['K'];uv=data['uv_pixels'];ids=data['object_id']
    wide=View([1.0,-.75,.65],[.08,-.06,.01],48)
    close=View([.37,-.29,.19],[.104,-.095,.06],35)
    scenes={kind:dict(ranges=data[key],history=np.empty((0,3)),writer=None) for kind,key in [('measured','range_m'),('ideal','ideal_range_m')]}
    if not args.preview:
        for kind,state in scenes.items():state['writer']=writer(OUT/('lidar_3d.mp4' if kind=='measured' else 'lidar_3d_ideal.mp4'))
    report=[];font=ImageFont.truetype(FONT,26);small=ImageFont.truetype(FONT,20)
    for i in range(450 if not args.preview else 145):
        for kind,state in scenes.items():
            ranges=state['ranges'][i];valid=np.isfinite(ranges);pixels=uv[i,valid];r=ranges[valid]
            direction=np.column_stack(((pixels[:,0]-K[0,2])/K[0,0],-(pixels[:,1]-K[1,2])/K[1,1],-np.ones(len(pixels))))
            direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            current=direction*r[:,None]@pose[:3,:3].T+pose[:3,3]
            static=current[ids[i,valid]==0][::8]
            # No dynamic history: do not leave ghost fingers after movement/release.
            history=state['history']
            if not args.preview or i==144:
                a=render(wide,history,current,r)
                b=render(close,history,current,r,True)
                # Clearly separated 3D overview and close-up; neither uses camera RGB.
                canvas=Image.new('RGB',(1920,800),(10,16,23))
                canvas.paste(a.resize((960,540),Image.Resampling.LANCZOS),(0,150))
                canvas.paste(b.resize((960,540),Image.Resampling.LANCZOS),(960,150))
                draw=ImageDraw.Draw(canvas)
                label='SIMULATED RANGE | 2 cm noise' if kind=='measured' else 'IDEAL RANGE | noise-free geometric reference'
                draw.text((24,20),f'{TRIAL.upper()} PRESS | 3D LiDAR point cloud',font=font,fill='white')
                draw.text((24,61),label,font=small,fill='#b9cfde')
                draw.text((1660,24),f't = {i/30:05.2f} s',font=font,fill='white')
                draw.text((24,111),'Third-person table view',font=font,fill='white')
                draw.text((984,111),'Bottle and hand detail | same point cloud',font=font,fill='white')
                draw.text((24,707),'Blue-gray: past static returns   |   Colored: current scan   |   Hand: current scan only',font=small,fill='#b9cfde')
                draw.text((24,746),'Display viewpoints differ from the fixed LiDAR pose; unseen surfaces are not invented.',font=small,fill='#b9cfde')
                bar=(matplotlib.colormaps['turbo'](np.linspace(0,1,240))[:,:3]*255).astype('uint8')
                canvas.paste(Image.fromarray(np.repeat(bar[None],14,axis=0)),(1620,727))
                draw.text((1560,750),'LiDAR range: 0.25 to 1.25 m',font=small,fill='white')
                if args.preview or i in [0,90,144,240]:canvas.save(OUT/f'lidar_3d_{kind}_{i:04d}.jpg',quality=95)
                if not args.preview:state['writer'].send(np.asarray(canvas))
            state['history']=np.concatenate([history,static])[-60000:]
        report.append(dict(frame=i,current_returns=int(np.isfinite(data['range_m'][i]).sum()),hand_returns=int((ids[i]==1).sum())))
        if i%60==0:print('3D POINT CLOUD',i,flush=True)
    if args.preview:return
    for state in scenes.values():state['writer'].close()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    (OUT/'lidar_3d.json').write_text(json.dumps(dict(source=str(source),source_sha256=source_hash,trial=TRIAL,
        videos=['lidar_3d.mp4','lidar_3d_ideal.mp4'],frames=450,fps=30,size=[1920,800],
        static_history='Every eighth static return; capped at 60000; muted blue-gray. No dynamic hand history.',
        viewer_eye_wide=wide.eye.tolist(),viewer_eye_close=close.eye.tolist(),
        sensor_pose_unchanged=True,no_rgb_background=True,color='Radial distance from the fixed LiDAR origin, fixed 0.25 to 1.25 m, turbo; static history brightened blue-gray.',
        color_source='sensor_radial_range_m',color_limits_m=[.25,1.25],
        display_point_radii_before_panel_resize=dict(current_overview=2,current_closeup=4,history_overview=1,history_closeup=2),
        history_color_rgb=[76,112,136],
        measured='Existing noisy ranges unchanged; no denoising or new geometry.',
        ideal='Existing ideal first-hit ranges, explicitly noise-free, identical scan directions.',
        limitations='Same close-range geometric scan approximation as lidar_visible.md. Both displays show only surfaces observed by the original fixed scanner; no unseen geometry filled in. Point size is a display setting.',
        reports=report),indent=2))
    print('3D POINT CLOUD COMPLETE',flush=True)

if __name__=='__main__':main()
