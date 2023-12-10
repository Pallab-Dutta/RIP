import joblib
import numpy as np
import MDAnalysis as mda
#import Res_Imp_Proj as RIP
import Res_Imp_Proj1 as RIP
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

def get_base_selstr(universe, capN=False, capC=False):
    selstr = 'protein'
    protein = universe.select_atoms(selstr)
    notRES = []
    if capN:
        startid = protein.residues.resids[0]
        notRES.append(f'resid {startid}')
    if capC:
        endid = protein.residues.resids[-1]
        notRES.append(f'resid {endid}')
    notRES = ' or '.join(notRES)
    selstr = f"({selstr} and not ({notRES}))"
    return selstr

u1 = mda.Universe("../src_act_npt_protein.tpr","../src_act_npt_500ns_pbcc.xtc")
u2 = mda.Universe("../src_int_npt_protein.tpr","../src_int_npt_500ns_pbcc.xtc")

#u1 = mda.Universe("../src_act_npt_protein.tpr","../src_act_npt_short.xtc")
#u2 = mda.Universe("../src_int_npt_protein.tpr","../src_int_npt_short.xtc")

Universes = {'active':u1, 'inactive':u2}

base_selstr = get_base_selstr(u1, capN=True, capC=True)
r4est = RandomForestClassifier(n_estimators = 5000, warm_start=True)
model = r4est

src_RIP = RIP.Residue_Importance_Projection([u1,u2], BaseSelStr=base_selstr, SelCodes=['dih-b'])
src_RIP.Train(model)
#model = src_RIP.model
src_RIP.Test()
joblib.dump(src_RIP.model, 'src_RIP.joblib')

src_RIP.get_FeatureImportance()
featureIMP = src_RIP.FeatureImportance
np.save("dih-b_featureIMP.npy",featureIMP)

src_RIP.project_ResidueImportance()
print('error table:\n',src_RIP.ErrorTable)

src_RIP.get_ImportantFeatures(u1, 5, 'act_dih-b_imp_features.npz')
src_RIP.get_ImportantFeatures(u2, 5, 'int_dih-b_imp_features.npz')

#src_RIP.save_RIP(PDBfile='../src_int_npt_lastframe.pdb', Name='inactive', viewer='pymol', OutFileHeader='int_dih-b')
#src_RIP.save_RIP(PDBfile='../src_int_npt_lastframe.pdb', viewer='nglview', OutFileHeader='int_dih-b')
#src_RIP.save_RIP(PDBfile='../src_act_npt_lastframe.pdb', Name='active', viewer='pymol', OutFileHeader='act_dih-b')
#src_RIP.save_RIP(PDBfile='../src_act_npt_lastframe.pdb', viewer='nglview', OutFileHeader='act_dih-b')
