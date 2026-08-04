import os, sys, shutil

SRC = '/workspace/core/source/amr/descriptions/turtlebot3'
DST = '/tmp/amr_test'

shutil.rmtree(DST, ignore_errors=True)
shutil.copytree(SRC, DST)

urdf_path = os.path.join(DST, 'model.urdf')
s = open(urdf_path).read()
s = s.replace('package://viro_amr_description/turtlebot3/meshes/', 'meshes/')
open(urdf_path, 'w').write(s)

print('URDF meshes now reference relative paths:')
print([l.strip() for l in open(urdf_path) if '<mesh' in l])
