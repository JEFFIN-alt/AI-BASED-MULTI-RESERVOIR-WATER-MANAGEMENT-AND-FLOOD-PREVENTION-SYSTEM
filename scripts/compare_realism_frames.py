"""Compare captured pixels along each projected channel, not shader uniforms."""
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/realism/final')
root=Path(parser.parse_args().root)
report=json.loads((root/'report.json').read_text(encoding='utf-8'))
comparison={}
sheet=Image.new('RGB',(1440,1040),'#eef0ed')
draw=ImageDraw.Draw(sheet)
for i in range(1,5):
    frames=report['links'][f'R{i}']
    path=frames[0]['diagnostics']['channels'][i-1]['screenPath']
    image0=Image.open(root/f'link{i}-t0.png').convert('RGB')
    # Restrict the comparison to actual channel pixels, excluding UI panels.
    mask=Image.new('L',image0.size)
    md=ImageDraw.Draw(mask)
    md.line([tuple(p) for p in path],fill=255,width=12)
    area=np.asarray(mask)>0
    a=np.asarray(image0).astype(np.int16)
    values=[]
    xs=[p[0] for p in path[:13]];ys=[p[1] for p in path[:13]]
    box=(max(0,int(min(xs)-22)),max(110,int(min(ys)-22)),min(image0.width,int(max(xs)+22)),min(image0.height-60,int(max(ys)+22)))
    for j,t in enumerate([0,1,2,4]):
        image=Image.open(root/f'link{i}-t{t}.png').convert('RGB')
        if t:
            delta=np.max(np.abs(np.asarray(image).astype(np.int16)-a),axis=2)
            values.append({'t':t,'channel_pixels_changed_gt_8':int(((delta>8)&area).sum()),'mean_channel_delta':float(delta[area].mean())})
        crop=image.crop(box)
        crop.thumbnail((350,230))
        sheet.paste(crop,(j*360,i*260-230))
        draw.text((j*360+8,i*260-255),f'R{i} outlet / channel | t={t}s',fill='#182e25')
    comparison[f'R{i}']={'crop':box,'pixel_comparisons':values}
    assert all(v['channel_pixels_changed_gt_8']>10 for v in values),comparison[f'R{i}']
sheet.save(root/'flow-contact-sheet.png')
(root/'pixel-comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
print(json.dumps(comparison,indent=2))
