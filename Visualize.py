import numpy as np
import MDAnalysis as mda
import sys
import matplotlib
from matplotlib import cm
import pickle
import pandas as pd
from tqdm.autonotebook import tqdm
from vscripts import *
#from FeatureResidue import *
#from FeatureImportance import *
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import SelectFromModel
#from sklearn.model_selection import train_test_split

def hexcolor(RGBcolors):
    HEXcolors = [matplotlib.colors.rgb2hex(color).upper() for color in RGBcolors]
    return HEXcolors

def saveobj(Object,FileName):
    FileID = open(FileName, 'wb')
    pickle.dump(Object, FileID)
    FileID.close()

def savetxt(Text,FileName):
    FileID = open(FileName,'w')
    print(Text, file=FileID)
    FileID.close()

def minmax(x):
    x_minmax = (x-min(x))/(max(x)-min(x))
    return x_minmax

class Visualize_Importance_Projection:
    def __init__(self, PDBfiles, SelStr, IMPfile='ResImpProj.npy'):
        self.proteins = PDBfiles
        self.selstr = SelStr
        self.IMP = np.load(IMPfile)

    def colorize(self, cmap='Spectral_r', OUTfile='colored_RIP.csv'):
        PDBfile = list(self.proteins.keys())[0]
        selstr = self.selstr
        protein = mda.Universe(PDBfile).select_atoms(selstr)
        resids = protein.residues.resids
        resnames = protein.residues.resnames
        colormap = cm.get_cmap(cmap)
        importance = self.IMP
        IMP_minmax = minmax(self.IMP)
        resRGBcolors = colormap(IMP_minmax)
        projection = hexcolor(resRGBcolors)
        IMP_data = {'resids':resids,'resnames':resnames,'importance':importance,'scaled_importance':IMP_minmax,'projection':projection}
        df = pd.DataFrame(IMP_data)
        self.IMP_data = df
        df.to_csv(OUTfile, index=False)

    def get_maxcontrib_resids(self, cutoff_imp=0.5):
        IMP = self.IMP
        IMP_sort_idx = np.argsort(IMP)[::-1]
        IMP_sort = IMP[IMP_sort_idx]
        IMP_cumsum = np.cumsum(IMP_sort)
        IMP_total = np.sum(IMP)
        IMP_cutoff = cutoff_imp*IMP_total
        IMP_cumidx = np.where(IMP_cumsum <= IMP_cutoff)[0]
        IMP_idx = IMP_sort_idx[:IMP_cumidx[-1] + 1]
        resids = self.IMP_data['resids'].to_numpy()
        IMP_resids = resids[IMP_idx]
        return IMP_resids

    def save_RIP(self, viewer, OutFileHeader, IMPfile='colored_RIP.csv', cutoff_imp=0.5):
        PDBfiles = self.proteins
        ressticks = list(self.get_maxcontrib_resids(cutoff_imp=cutoff_imp))
        resids = self.IMP_data['resids'].tolist()
        rescolors = self.IMP_data['projection'].tolist()
        rescolors = [color.replace('#','0x') for color in rescolors]
        if viewer.lower() in ['pymol','pymol_basic']:
            script = pymol_basic.format(PDBfiles=PDBfiles,ressticks=ressticks,resids=resids,rescolors=rescolors)
        if viewer.lower() == 'pymol_advanced':
            script = pymol_advanced.format(PDBfiles=PDBfiles,IMPfile=IMPfile)
        if viewer.lower() == 'chimerax':
            script = chimeraX.format(PDBfiles=PDBfiles,IMPfile=IMPfile)
        if viewer == 'nglview':
            rescolors = list(self.Projection.values())
            resids = self.Resids4Projection + delta_resid
            numres = len(resids)
            colorscheme = [[hexcap(rescolors[i]),str(resids[i])] for i in range(numres)]
            colorFile = OutFileHeader+'_colorscheme_nglview.list'
            saveobj(colorscheme, colorFile)
            Str = f"""# Residue Importance Projection (RIP) for nglview

import nglview as nv
import pickle

FileID = open('{colorFile}','rb')
colorscheme = pickle.load(FileID)
FileID.close()

PDBfile = '{PDBfile}'
view = nv.NGLWidget()
view.add_component(PDBfile)
nv.color.ColormakerRegistry.add_selection_scheme(
    "rip_scheme", colorscheme
)
view.clear()
view.add_cartoon(color="rip_scheme")
view
"""
            OUTfile = OutFileHeader+'_nglview.py'
            fid = open(OUTfile,'w')
            print(Str, file=fid)
            fid.close()
        OUTfile = OutFileHeader+'_'+viewer.lower()+'.py'
        fid = open(OUTfile,'w')
        print(script, file=fid)
        fid.close()

def main():
    args = sys.argv[1:]
    parm, traj, slcd = args
    u = mda.Universe(parm, traj)
    features = extract_features(universe=u, selcode=slcd, fs=0, fe=24)
    print(features,features.shape)

if __name__ == '__main__':
    main()
