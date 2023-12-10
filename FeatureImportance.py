import numpy as np
import MDAnalysis as mda
from FeatureResidue import *

# Get Secondary Feature Importance

def get_dih_imp(AtomGroup, feature_importance, not_required=True):
    fimp = feature_importance.reshape(2,-1)
    dih_imp = fimp.sum(axis=0)
    print('get_dih_imp')
    print(dih_imp.min(),dih_imp.max())
    return dih_imp

def get_dis_imp(AtomGroup, feature_importance, half_triangle=False):
    fimp = feature_importance
    numAtoms = len(AtomGroup)
    dismat_imp = np.zeros((numAtoms,numAtoms))
    for i in range(numAtoms):
        j=i+1
        dismat_imp_ij = fimp[int(i/2*(2*(numAtoms-i)+(i-1)*1)):int(j/2*(2*(numAtoms-j)+(j-1)*1))]
        dismat_imp[i,i+1:numAtoms] = dismat_imp_ij
        if (not half_triangle):
            dismat_imp[i+1:numAtoms,i] = dismat_imp_ij
    print('get_dis_imp')
    print(np.min(dismat_imp),np.max(dismat_imp))
    return dismat_imp



# Get Residue (Backbone and Sidechain) Importance

def get_resid_imp(baseAtomGroup, AtomGroup, secnd_feature_imp, secnd_feature_type):
    backbone_ag = AtomGroup.select_atoms('name CA')
    sidechain_ag = AtomGroup.select_atoms('name CB')
    backbone_bool = AtomGroup.names == 'CA'
    sidechain_bool = AtomGroup.names == 'CB'
    numres_backbone = sum(backbone_bool)
    numres_sidechain = sum(sidechain_bool)
    Backbone = np.zeros(numres_backbone)
    Sidechain = np.zeros(numres_sidechain)
    if secnd_feature_type==0:
        sfi = secnd_feature_imp.reshape(2,-1)
        #phiimp1 = np.append(np.delete(sfi[0],0),0)*0.25
        #phiimp2 = sfi[0]*0.75
        #psiimp1 = sfi[1]*0.75
        #psiimp2 = np.insert(np.delete(sfi[1],-1),0,0)*0.25
        #resimp = phiimp1+phiimp2+psiimp1+psiimp2
        resimp = sfi.max(axis=0)
        Backbone += resimp
    elif secnd_feature_type==1:
        sfi = secnd_feature_imp.reshape(2,-1)
        resimp = sfi.max(axis=0)
        Sidechain += resimp
    elif secnd_feature_type==2:
        all_ag = backbone_ag+sidechain_ag
        backbone_bool = all_ag.names == 'CA'
        sidechain_bool = all_ag.names == 'CB'
        posimp_backbone = secnd_feature_imp.max(axis=0)[backbone_bool]		# one may use sum operation instead
        posimp_sidechain = secnd_feature_imp.max(axis=0)[sidechain_bool]	# one may use sum operation instead
        if 'CA' in AtomGroup.names:
            resIDs = backbone_ag.resids.reshape(-1,1)
            _,uid = np.unique(resIDs==resIDs.T,axis=0,return_index=True)
            binuid = (resIDs==resIDs.T)[np.sort(uid),:]
            resimp = np.max(posimp_backbone*binuid,axis=1)			# one may use sum operation instead
            Backbone += resimp
        if 'CB' in AtomGroup.names:
            resIDs = sidechain_ag.resids.reshape(-1,1)
            _,uid = np.unique(resIDs==resIDs.T,axis=0,return_index=True)
            binuid = (resIDs==resIDs.T)[np.sort(uid),:]
            resimp = np.max(posimp_sidechain*binuid,axis=1)			# one may use sum operation instead
            Sidechain += resimp
    return Backbone, Sidechain


def get_imp_dih_args(Universe, AtomGroup, selcode, secnd_feature_imp, secnd_feature_type, number):
    sfi = secnd_feature_imp.reshape(2,-1)
    maxids = np.argsort(sfi.flatten())[::-1]
    ids = maxids[:number]
    del maxids
    args = []
    ags = []
    print('ids',ids)
    for idx in ids:
        dih,res = np.unravel_index(idx, sfi.shape)
        print(dih,res)
        print(dih.shape,res.shape,AtomGroup.resids.shape)
        resid = AtomGroup.resids[res]
        print(resid)
        selstr, arg_selcode, get_feature_vals = get_ImpFeatureDictionary(resid)[selcode]
        ags.append(Universe.select_atoms(selstr))
        args.append([arg_selcode,dih])
    dih_args = [ags, args]
    return dih_args, get_feature_vals

def get_imp_dis_args(Universe, AtomGroup, selcode, secnd_feature_imp, secnd_feature_type, number):
    maxids = np.argsort(secnd_feature_imp.flatten())[::-1]
    ids = maxids[:number]
    del maxids
    ag_pairs = []
    arg_pairs = []
    CAB = {'CA':0,'CB':1}
    ag1 = AtomGroup.select_atoms('name CA')
    ag2 = AtomGroup.select_atoms('name CB')
    AtomGroup = ag1+ag2
    for idx in ids:
        res1,res2 = np.unravel_index(idx, secnd_feature_imp.shape)
        resid1 = AtomGroup.resids[res1]
        resid2 = AtomGroup.resids[res2]
        name1 = AtomGroup.names[res1]
        name2 = AtomGroup.names[res2]
        selstr1, arg_selcode1, get_feature_vals = get_ImpFeatureDictionary(resid1)[selcode]
        selstr2, arg_selcode2, get_feature_vals = get_ImpFeatureDictionary(resid2)[selcode]
        ag1 = Universe.select_atoms(selstr1[CAB[name1]])
        ag2 = Universe.select_atoms(selstr2[CAB[name2]])
        ag_pairs.append([ag1,ag2])
        arg1 = arg_selcode1[CAB[name1]]
        arg2 = arg_selcode2[CAB[name2]]
        arg_pairs.append([arg1,arg2])
    dis_args = [ag_pairs, arg_pairs, 2]
    print(dis_args)
    return dis_args, get_feature_vals


sidechain_dict_4com = {
    "GLY" : "(resname GLY and name CB)",
    "ALA" : "(resname ALA and name CB)",
    "VAL" : "(resname VAL and name CG*)",
    "CYS" : "(resname CYS and name SG)",
    "PRO" : "(resname PRO and name CG)",
    "LEU" : "(resname LEU and name CD*)",
    "ILE" : "(resname ILE and name CD)",
    "MET" : "(resname MET and name CE)",
    "TRP" : "(resname TRP and (name CG or name CD* or name CE* or name CZ* or name CH* or name NE*))",
    "PHE" : "(resname PHE and (name CG or name CD* or name CE* or name CZ))",
    "LYS" : "(resname LYS and name NZ)",
    "ARG" : "(resname ARG and name NH*)",
    "HIS" : "(resname HIS and (name CG* or name CD* or name ND* or name CE* or name NE*))",
    "SER" : "(resname SER and name OG)",
    "THR" : "(resname THR and (name OG* or name CG*))",
    "TYR" : "(resname TYR and (name CG or name CD* or name CE* or name CZ or name OH*))",
    "ASN" : "(resname ASN and (name CG or name ND* or name OD*))",
    "GLN" : "(resname GLN and (name CD or name NE* or name OE*))",
    "ASP" : "(resname ASP and (name CG or name OD*))",
    "GLU" : "(resname GLU and (name CD or name OE*))"
}

def get_ImpFeatureDictionary(resid):
    base_selstr = f'(protein and resid {resid})'
    selcodes = ['dih-b','dih-s','dis-b','dis-s','dis-ca','dis-com','dis-bs','dis-cacom']
    arg_selcodes = ['rama', 'janin', ['pos']*2, ['pos']*2, ['pos']*2, ['com']*2, ['pos']*2, ['pos','com']]
    sidesel = ' or '.join(list(sidechain_dict_4com.values()))
    selstrs = [f'{base_selstr}', \
               f'{base_selstr} and not (resname ALA CYS* GLY PRO SER THR VAL)', \
               [f'{base_selstr} and backbone']*2, \
               [f'{base_selstr} and ((name C* or name O* or name N* or name S*) and not backbone)']*2, \
               [f'{base_selstr} and name CA']*2, \
               [f'{base_selstr} and ({sidesel})']*2, \
               [f'{base_selstr} and (name C* or name O* or name N* or name S*)']*2, \
               [f'{base_selstr} and name CA', f'{base_selstr} and ({sidesel})']]
    get_feature_vals = [Dihedral]*2 + [Distance]*6
    numcodes = len(selcodes)
    ImpFeatureDict = {}
    for i in range(numcodes):
        ImpFeatureDict[selcodes[i]] = [selstrs[i], arg_selcodes[i], get_feature_vals[i]] 
    return ImpFeatureDict

def get_ImportanceDictionary(base_selstr):
    selcodes = ['dih-b','dih-s','dis-b','dis-s','dis-ca','dis-com','dis-bs','dis-cacom']
    repstrs = [f'{base_selstr} and name CA', \
               f'({base_selstr} and name CB) and not (resname ALA CYS* GLY PRO SER THR VAL)', \
               f'{base_selstr} and name CA', \
               f'{base_selstr} and name CB', \
               f'{base_selstr} and name CA', \
               f'{base_selstr} and name CB', \
               f'{base_selstr} and (name CA or name CB)', \
               f'{base_selstr} and (name CA or name CB)']
    get_secnd_imp = [get_dih_imp]*2 + [get_dis_imp]*6
    secnd_feature_type = [0,1] + [2]*6
    get_imp_feature_farg = [get_imp_dih_args]*2 + [get_imp_dis_args]*6
    numcodes = len(selcodes)
    ImportanceDict = {}
    for i in range(numcodes):
        ImportanceDict[selcodes[i]] = [repstrs[i], get_secnd_imp[i], secnd_feature_type[i], get_imp_feature_farg[i]]
    return ImportanceDict

def get_ResidueImportance(Universe, BaseSelstr, SelCodes, FeatureImp, FeatureIDs):
    ImportanceDict = get_ImportanceDictionary(BaseSelstr)
    base_ag = Universe.select_atoms(BaseSelstr)
    base_resids = base_ag.residues.resids
    numres = len(base_resids)
    Dih_backbone = np.zeros(numres) 
    Dih_sidechain = np.zeros(numres) 
    Dis_backbone = np.zeros(numres)
    Dis_sidechain = np.zeros(numres)
    for selcode in SelCodes:
        repstr, get_secnd_imp, secnd_feature_type, _ = ImportanceDict[selcode]
        ag = Universe.select_atoms(repstr)
        ag_backbone = ag.select_atoms('name CA')
        ag_sidechain = ag.select_atoms('name CB')
        resids_backbone = ag_backbone.resids.reshape(-1,1)
        resids_sidechain = ag_sidechain.resids.reshape(-1,1)
        impids_backbone = np.sum(base_resids==resids_backbone,axis=0,dtype=np.bool8)
        impids_sidechain = np.sum(base_resids==resids_sidechain,axis=0,dtype=np.bool8)
        FeatureIDX = FeatureIDs==secnd_feature_type
        print('get_ResidueImportance')
        print(np.min(FeatureImp[FeatureIDX]), np.max(FeatureImp[FeatureIDX]))
        secnd_imp = get_secnd_imp(ag, FeatureImp[FeatureIDX])
        resid_imp = get_resid_imp(base_ag, ag, secnd_imp, secnd_feature_type)
        if secnd_feature_type==2:
            Dis_backbone[impids_backbone] += resid_imp[0]
            Dis_sidechain[impids_sidechain] += resid_imp[1]
        elif secnd_feature_type==0:
            Dih_backbone[impids_backbone] += resid_imp[0]
        elif secnd_feature_type==1:
            Dih_sidechain[impids_sidechain] += resid_imp[1]
    All_backbone = Dis_backbone + Dih_backbone
    All_sidechain = Dis_sidechain + Dih_sidechain
    AllImportance = All_backbone + All_sidechain
    ResidueImportance = {'backbone' : {'all':All_backbone, 'dis':Dis_backbone, 'dih':Dih_backbone },\
                         'sidechain': {'all':All_sidechain,'dis':Dis_sidechain,'dih':Dih_sidechain}}
    return AllImportance, ResidueImportance

def get_important_residues(Universe, BaseSelstr, SelCodes, FeatureImp, FeatureIDs, number):
    important_features = {}
    DihDone = False
    DisDone = False
    ImportanceDict = get_ImportanceDictionary(BaseSelstr)
    for selcode in SelCodes:
        repstr, get_secnd_imp, secnd_feature_type, get_imp_feature_farg = ImportanceDict[selcode]
        ag = Universe.select_atoms(repstr)
        FeatureIDX = FeatureIDs==secnd_feature_type
        secnd_imp = get_secnd_imp(ag, FeatureImp[FeatureIDX], True)
        imp_args, imp_func = get_imp_feature_farg(Universe, ag, selcode, secnd_imp, secnd_feature_type, number)
        imp_obj = imp_func(*imp_args)
        imp_obj.run()
        important_features[selcode] = imp_obj.results
    return important_features
