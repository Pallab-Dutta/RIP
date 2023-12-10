pymol_basic = \
"""

# Residue Importance Projection (RIP) for pymol #

from pymol import cmd
from pymol.cgo import *


# Load data

PDBfiles = {PDBfiles}
Names = list(PDBfiles.values())


# Load structures in viewer

cmd.bg_color('white')

for PDBfile,Name in PDBfiles.items():
    cmd.load('%s'%(PDBfile),'%s'%(Name))

for Name in Names[1:]:
    cmd.align('%s'%(Name),'%s'%(Names[0])) 


# Highlight residues with importance 

resSticks = {ressticks}
for Name in Names:
    for res in resSticks:
        cmd.show('licorice','%s and resi %d'%(Name,res))


# Color residues according to importance

resIDs = {resids}
resColors = {rescolors}
for Name in Names:
    for res,color in zip(resIDs,resColors):
        cmd.color(color,'%s and resi %d'%(Name,res))

"""

pymol_advanced = \
"""
# Residue Importance Projection (RIP) for pymol #


# Import required modules

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm
from pymol import cmd
from pymol.cgo import *
from config import *


# Load data

PDBfiles = {PDBfiles}
Names = list(PDBfiles.values())
IMPdf = pd.read_csv('{IMPfile}')
resids = IMPdf['resids'].to_numpy()
resimp = IMPdf['importance'].to_numpy()
sclimp = IMPdf['scaled_importance'].to_numpy()


# Generate colors

def hexcolor(RGBcolors):
    HEXcolors = [matplotlib.colors.rgb2hex(color).replace('#','0x') for color in RGBcolors]
    return HEXcolors

colormap = cm.get_cmap(cmap)		# get cmap value from config.py
resRGBcolors = colormap(sclimp)
rescolors = hexcolor(resRGBcolors)


# Load structures in viewer

cmd.bg_color('white')

for PDBfile,Name in PDBfiles.items():
    cmd.load('%s'%(PDBfile),'%s'%(Name))

for Name in Names[1:]:
    cmd.align('%s'%(Name),'%s'%(Names[0])) 


# get most important resids that collectively contributes
# a fraction of the total importance. By default we set this fraction to 0.5

def get_maxcontrib_resids(RESIDs, IMP, fraction=0.5):
    IMP_sort_idx = np.argsort(IMP)[::-1]
    IMP_sort = IMP[IMP_sort_idx]
    IMP_cumsum = np.cumsum(IMP_sort)
    IMP_total = np.sum(IMP)
    IMP_cutoff = IMP_total*fraction
    IMP_cumidx = np.where(IMP_cumsum <= IMP_cutoff)[0]
    IMP_idx = IMP_sort_idx[:IMP_cumidx[-1] + 1]
    IMP_resids = RESIDs[IMP_idx]
    return IMP_resids

resSticks = get_maxcontrib_resids(resids,resimp,frac)		# get frac value from config.py


# Highlight the selected residues in stick representation

for Name in Names:
    for res in resSticks:
        cmd.show('licorice','%s and resi %d'%(Name,res))


# Color residues according to importance

for Name in Names:
    for res,color in zip(resids,rescolors):
        cmd.color(color,'%s and resi %d'%(Name,res))

"""

chimeraX = \
"""

# Residue Importance Projection (RIP) for chimeraX

import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm
from chimerax.core.commands import run
from config import *


# Load data

PDBfiles = {PDBfiles}
PDBfiles = list(PDBfiles.keys())
IMPdf = pd.read_csv('{IMPfile}')
resids = IMPdf['resids'].to_numpy()
resimp = IMPdf['importance'].to_numpy()
sclimp = IMPdf['scaled_importance'].to_numpy()


# Generate colors

def hexcolor(RGBcolors):
    HEXcolors = [matplotlib.colors.rgb2hex(color).upper() for color in RGBcolors]
    return HEXcolors

colormap = cm.get_cmap(cmap)            # get cmap value from config.py
resRGBcolors = colormap(sclimp)
rescolors = hexcolor(resRGBcolors)


# Load structures in viewer

models = []
for i,PDBfile in enumerate(PDBfiles):
    run(session, "open %s"%(PDBfile))
    modelNo = i+1
    models.append(modelNo)

for model in models[1:]:
    run(session, "align #%d to #%d"%(model,models[0]))


# get most important resids that collectively contributes
# a fraction of the total importance. By default we set this fraction to 0.5

def get_maxcontrib_resids(RESIDs, IMP, fraction=0.5):
    IMP_sort_idx = np.argsort(IMP)[::-1]
    IMP_sort = IMP[IMP_sort_idx]
    IMP_cumsum = np.cumsum(IMP_sort)
    IMP_total = np.sum(IMP)
    IMP_cutoff = IMP_total*fraction
    IMP_cumidx = np.where(IMP_cumsum <= IMP_cutoff)[0]
    IMP_idx = IMP_sort_idx[:IMP_cumidx[-1] + 1]
    IMP_resids = RESIDs[IMP_idx]
    return IMP_resids

resSticks = get_maxcontrib_resids(resids,resimp,frac)           # get frac value from config.py


# Highlight the selected residues in stick representation

for model in models:
    for res in resSticks:
        run(session, "show #%d:%d atoms"%(model,res))


# Color residues according to importance

for model in models:
    for res,color in zip(resids,rescolors):
        run(session, "color #%d:%d %s"%(model,res,color))


run(session, "save OBJ.glb textureColors true")

"""
