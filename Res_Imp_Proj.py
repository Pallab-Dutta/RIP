import numpy as np
import MDAnalysis as mda
import sys
import matplotlib
from matplotlib import cm
import pickle
import pandas as pd
from tqdm.autonotebook import tqdm
from FeatureResidue import *
from FeatureImportance import *
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import SelectFromModel
#from sklearn.model_selection import train_test_split

def hexcap(hexstr):
    head = '#'
    tail = hexstr[2:].upper()
    hexstrCAP = head+tail
    return hexstrCAP

def saveobj(Object,FileName):
    FileID = open(FileName, 'wb')
    pickle.dump(Object, FileID)
    FileID.close()

def savetxt(Text,FileName):
    FileID = open(FileName,'w')
    print(Text, file=FileID)
    FileID.close()

class Residue_Importance_Projection:
    def __init__(self, Universes, BaseSelStr, SelCodes):
        self.Universes = Universes
        self.classes = np.arange(0,len(Universes),1,dtype=np.int32)
        self.BaseSelStr = BaseSelStr
        self.AtomGroup = None
        self.FeatureImportance = None
        self.ResidueImportance = None
        self.Projection = None
        if len(SelCodes)>2:
            raise ValueError("SelCodes must contain not more than 2 elements")
        #elif all('dih' in selcode for selcode in SelCodes):
        #    raise ValueError("Two different types of selcodes are only allowed, e.g. ['dih-b','dis-ca']")
        elif 'dih-bs' in SelCodes:
            dih_bs = ['dih-b','dih-s']
            self.SelCodes = dih_bs
            idx = SelCodes.index('dih-bs')
            #if idx==0:
            #    self.SelCodes = ['dih-b','dih-s']+[SelCodes[1]]
            #if idx==1:
            #    self.SelCodes = [SelCodes[0]]+['dih-b','dih-s']
            if len(SelCodes)==2:
                if idx==0:
                    self.SelCodes = dih_bs+[SelCodes[1]]
                if idx==1:
                    self.SelCodes = [SelCodes[0]]+dih_bs
        else:
            self.SelCodes = SelCodes

    def TrainTestSplit(self, Train_size=0.9):
        univ0 = self.Universes[0]
        self.Resids4Projection = univ0.select_atoms(self.BaseSelStr).residues.resids
        TotalFrames = len(univ0.trajectory)
        TrainFrames = round(TotalFrames*Train_size)
        self.TrainFrames = TrainFrames

    def LabelBatchData(self, Data, Label):
        RowLen = Data.shape[0]
        Labels = np.zeros((RowLen,1))+Label
        LabeledData = np.hstack((Data,Labels))
        return LabeledData

    def Train(self, model):
        self.model = model
        try:
            TrnFrames = self.TrainFrames
        except:
            self.TrainTestSplit()
            TrnFrames = self.TrainFrames
        Features = []
        print("\nPrepairing Data for training...\n")
        outNames = ['./act_train_features.nc','./int_train_features.nc']
        for i,cls in enumerate(self.classes):
            u = self.Universes[cls]
            feature, featureIDs = get_FeatureValues(u, self.BaseSelStr, self.SelCodes, stop=TrnFrames, outFile=outNames[i])
            feature = self.LabelBatchData(feature, cls)
            Features.append(feature)
            del feature
        Features = np.vstack(Features)
        np.random.shuffle(Features)
        print("\nTraining started...\n")
        self.model.fit(Features[:,:-1],Features[:,-1])
        del Features
        self.FeatureIDs = featureIDs

    def Test(self):
        numClass = len(self.classes)
        errors = np.zeros((numClass,numClass))
        try:
            TrnFrames = self.TrainFrames
        except:
            self.TrainTestSplit()
            TrnFrames = self.TrainFrames
        Features = []
        print("\nPrepairing Data for testing...\n")
        outNames = ['./act_test_features.nc','./int_test_features.nc']
        for i,cls in enumerate(self.classes):
            u = self.Universes[cls]
            feature, featureIDs = get_FeatureValues(u, self.BaseSelStr, self.SelCodes, start=TrnFrames, outFile=outNames[i])
            feature = self.LabelBatchData(feature, cls)
            Features.append(feature)
            del feature
        Features = np.vstack(Features)
        np.random.shuffle(Features)
        X_test = Features[:,:-1]
        Y_test = Features[:,-1]
        del Features
        print("\nTesting started...\n")
        Y_pred = self.model.predict(X_test)
        Err = pd.crosstab(Y_test,Y_pred).to_numpy()
        errors += Err
        del Y_pred
        del X_test, Y_test
        percent_errors = errors/np.sum(errors)
        self.ErrorTable = percent_errors

    def get_FeatureImportance(self):    
        self.FeatureImportance = self.model.feature_importances_

    def get_ResidueImportance(self):
        self.get_FeatureImportance()
        u = self.Universes[0]
        featureIMP = self.FeatureImportance
        featureIDs = self.FeatureIDs
        residueIMP = get_ResidueImportance(u, self.BaseSelStr, self.SelCodes, featureIMP, featureIDs)
        self.ResidueImportance = residueIMP

    def get_ImportantFeatures(self, universe, number, outfile):
        featureIMP = self.FeatureImportance
        featureIDs = self.FeatureIDs
        ImpFeatures_Dict = get_important_residues(universe, self.BaseSelStr, self.SelCodes, featureIMP, featureIDs, number)
        np.savez(outfile, **ImpFeatures_Dict)

    def project_ResidueImportance(self, OUTfile='ResImpProj.npy'):
        self.get_ResidueImportance()
        IMP,_ = self.ResidueImportance
        IMP = IMP/IMP.sum()
        self.IMP = IMP
        np.save(OUTfile,self.IMP)

        #IMP_minmax = (IMP-IMP.min())/(IMP.max()-IMP.min())
        #self.IMP_minmax = IMP_minmax
        #cmap = cm.get_cmap('Spectral_r')   # colormap Spectral
        #resRGBcolors = cmap(IMP_minmax)
        #rescolors = [matplotlib.colors.rgb2hex(color).replace('#','0x') for color in resRGBcolors]  # convert to hex colors
        #resids = self.Resids4Projection
        #self.Projection = {resids[i]:rescolors[i] for i in range(len(resids))}
        #saveobj(self.Projection, OUTfile)

    def get_maxcontrib_resids(self, cutoff_imp=0.5):
        IMP = self.IMP
        IMP_sort_idx = np.argsort(IMP)[::-1]
        IMP_sort = IMP[IMP_sort_idx]
        IMP_cumsum = np.cumsum(IMP_sort)
        IMP_total = np.sum(IMP)
        IMP_cutoff = cutoff_imp*IMP_total
        IMP_cumidx = np.where(IMP_cumsum <= IMP_cutoff)[0]
        IMP_idx = IMP_sort_idx[:IMP_cumidx[-1] + 1]
        return IMP_idx

    def save_RIP(self, PDBfiles, selstr, viewer, OutFileHeader, cutoff_imp=0.5):
        IMP = self.IMP
        IMP_minmax = self.IMP_minmax
        univ0 = mda.Universe(PDBfiles[0])
        sel = univ0.select_atoms(f'{selstr}')
        resids = sel.residues.resids
        resnames = sel.residues.resnames
        data = {'resids':resids,'resnames':resnames,'importance':IMP_L_minmax}
        df = pd.DataFrame(data)
        csvout = OutFileHeader+'.csv'
        df.to_csv(csvout)
        if viewer.lower() == 'pymol':
            OUTfile = OutFileHeader+'_pymol.py'
            headStr = f"""# Residue Importance Projection (RIP) for pymol

import pandas as pd
from pymol import cmd
from pymol.cgo import *

cmd.bg_color('white')
PDBfiles = {PDBfiles}
Names = list(PDBfiles.values())
for PDBfile,Name in PDBfiles.items():
    cmd.load('{PDBfile}', '{Name}')
Name0 = Names[0]
for Name in Names[1:]:
    cmd.align({Name},{Name0})

cmd.show('cartoon')
obj=[]
df = pd.read_csv('{csvout}')

# Highlight residues with importance > {cutoff_imp}

"""

    def save_RIP_old(self, PDBfile, viewer, OutFileHeader, Name='protein', cutoff_imp=0.5, scale='L'):
        IMPfile = OutFileHeader+'_ResidueImportance.dict'
        IMP = self.ResidueImportance
        saveobj(IMP, IMPfile)
        #np.save(IMPfile, IMP)
        univ0 = mda.Universe(PDBfile)
        if scale == 'L':
            IMP_minmax = self.IMP_L_minmax
        elif scale == 'LL':
            IMP_minmax = self.IMP_LL_minmax
        #IDX_to_highlight = IMP_minmax > cutoff_imp
        IDX_to_highlight = self.get_maxcontrib_resids(scale=scale, cutoff_imp=0.5)
        AtomGroup = univ0.select_atoms('protein')
        delta_resid = min(AtomGroup.resids)-1
        if viewer == 'pymol':
            OUTfile = OutFileHeader+'_pymol.py'
            projection = self.Projection
            resids = self.Resids4Projection
            resids_to_highlight = resids[IDX_to_highlight]
            numres = len(resids)
            numres_to_highlight = len(resids_to_highlight)
            headStr = f"""# Residue Importance Projection (RIP) for pymol

from pymol import cmd
from pymol.cgo import *
cmd.bg_color('white')
cmd.load('{PDBfile}', '{Name}')
cmd.show('cartoon')
obj=[]


# Highlight residues with importance > {cutoff_imp}

"""
            colorStr = "cmd.color('%s', 'resi %d')\n"
            representationStr = "cmd.show('licorice', 'resi %d')\n"
            taleStr = "\ncmd.load_cgo(obj, 'cor_-06_-04')"
            Str = headStr
            for i in range(numres_to_highlight):
                refid_to_highlight = resids_to_highlight[i]
                resid_to_highlight = refid_to_highlight + delta_resid
                midStr = representationStr%(resid_to_highlight)
                Str = Str+midStr
            midStr = "\n\n# Color the representation according to RIP\n"
            Str = Str+midStr
            for i in range(numres):
                refid = resids[i]
                resid = refid + delta_resid
                rescolor = projection[refid]
                midStr = colorStr%(rescolor,resid)
                Str = Str+midStr
            fid = open(OUTfile,'w')
            print(Str, file=fid)
            fid.close()
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

def main():
    args = sys.argv[1:]
    parm, traj, slcd = args
    u = mda.Universe(parm, traj)
    features = extract_features(universe=u, selcode=slcd, fs=0, fe=24)
    print(features,features.shape)

if __name__ == '__main__':
    main()
