from pathlib import Path

p = Path('run_length_hazard.py')
text = p.read_text(encoding='utf-8')
old = 'LENGTH_SMOOTH_BLEND_MIN = 0.18\nLENGTH_SMOOTH_BLEND_MAX = 0.42'
new = 'LENGTH_SMOOTH_BLEND_MIN = 0.24\nLENGTH_SMOOTH_BLEND_MAX = 0.48'
if old not in text:
    raise SystemExit('post-patch length blend constants not found')
text = text.replace(old, new, 1)
old = '            0.25,\n            0.85,\n        )'
new = '            0.25,\n            0.72,\n        )'
if old not in text:
    raise SystemExit('context specificity clip not found')
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print('follow-up hazard smoothing applied')
