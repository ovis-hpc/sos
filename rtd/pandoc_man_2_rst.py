#/usr/bin/python3
#requires pandoc v3 for man input format
#point X_ROOT to the build directory

import os
import glob

SOS_ROOT='/opt/ovis/build/sos'
source = ['rpc','sos/python']
dest = ['sos_man','sos_man']
for c,s in enumerate(source):
    files = glob.glob(f'{SOS_ROOT}/{s}/*man')
    for i in files:
        fname = i.split('/')[-1].replace('.man','.rst')
        os.system('mkdir -p man2rst/')
        os.system(f'/usr/local/bin/pandoc -f man -s -t rst --toc {i} -o man2rst/{fname}')
        plugin = fname.replace('.rst','')
        plugin_title = '='*len(plugin)
        os.system('sed -i -e "0,/man/{s/man/'+plugin+'/}" man2rst/'+fname) 
        os.system('sed -i -e "s/===/'+plugin_title+'/" man2rst/'+fname) 
        os.system(f'cp man2rst/{fname} docs/source/{dest[c]}') 

