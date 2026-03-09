import shutil
import os

src = 'temp_tabdiff/tabdiff'
dst = 'src/baselines/tabdiff_core'
if not os.path.exists(dst):
    os.makedirs(dst)

for root, dirs, files in os.walk(src):
    for d in dirs:
        os.makedirs(os.path.join(dst, os.path.relpath(os.path.join(root, d), src)), exist_ok=True)
    for f in files:
        if f.endswith('.py') or f.endswith('.yaml'):
            src_file = os.path.join(root, f)
            dst_file = os.path.join(dst, os.path.relpath(src_file, src))
            shutil.copy2(src_file, dst_file)
            
# Create empty init file for package imports
with open(os.path.join(dst, '__init__.py'), 'w') as f:
    f.write('')
