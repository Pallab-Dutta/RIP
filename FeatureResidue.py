import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances
import netCDF4 as nc


# pramary features

def get_INP(AtomGroup):
    return AtomGroup

def get_POS(AtomGroup):
    POSs = AtomGroup.positions
    return POSs

def get_COM(AtomGroup, group='residue'):
    AtomGroup_array = AtomGroup.split(group)
    COMs = np.array([ag.center_of_mass() for ag in AtomGroup_array], dtype=np.float32).reshape(-1,3)
    return COMs



# secondary features

class Distance(AnalysisBase):
    """Calculate distances among specified atomgroups.

    Distances will be calculated between two atomgroups that are given for
    each step in the trajectory. Each :class:`~MDAnalysis.core.groups.AtomGroup`
    must contain at least 1 atom.

    Note
    ----
    This class takes a list as an input and is most useful for a large
    selection of atomgroups. If there is only one atomgroup of interest, then
    it must be given as a list of one atomgroup.

    """

    def __init__(self, atomgroup_pairs, typecoord_pairs, pairwise=0, **kwargs):
        """Parameters
        ----------
        atomgroup_pairs : list
            Distances between each atomgroup pairs are calculated.

        typecoord_pairs : list
            Positions from each atomgroup are calculated according to typecoord.
            When typecoord is 'pos' it calculates positions, and when typecoord
            is 'com' it calculates residue-wise center of masses.

        pairwise : int or list
            Determines how the distances between first and second set of coordinates
            are calculated. When pairwise value is 0 (default), first and second set 
            of coordinates are combined and distances among all possible position pairs 
            from the combined set are calculated. If pairwise value is 1, distances 
            between all possible pairs of first and second set of coordinates are 
            calculated. In case of pairwise value 2, distances between first and second
            set of coordinates are calculated, considering each row as a pair. In this
            case, first and second set must have same shape.

        Raises
        ------
        ValueError
            1. If length of any of the AtomGroup pair-lists is not 2.
            2. If any atomgroups do not contain at least 1 atom.

        """
        super(Distance, self).__init__(
            atomgroup_pairs[0][0].universe.trajectory, **kwargs)
        self.ag_pairs = atomgroup_pairs
        self.tc_pairs = typecoord_pairs
        coord_types = ['pos','com']
        self.get_coord = {'pos': get_POS, 'com': get_COM}
        if pairwise in [0,1,2]:
            self.pairwise = [pairwise]*len(self.ag_pairs)
        elif type(pairwise) == type(self.ag_pairs):
            if any([pw not in [0,1,2] for pw in pairwise]):
                raise ValueError(f"Pairwise must be an int among or a list containing: 0,1,2")
            if len(pairwise) == len(self.ag_pairs):
                self.pairwise = pairwise
            else:
                raise ValueError(f"Length of pairwise-list must be equal to the length of atomgroup_pairs.")

        if any([len(ag_pair) != 2 for ag_pair in atomgroup_pairs]):
            raise ValueError("Each AtomGroup pair must contain 2 AtomGroups")
        if any([len(ag_pair[0]) < 1 for ag_pair in atomgroup_pairs]) or any([len(ag_pair[1]) < 1 for ag_pair in atomgroup_pairs]):
            raise ValueError("Each AtomGroup must contain at least 1 atom")
        if len(typecoord_pairs) != len(atomgroup_pairs):
            raise ValueError("Length of atomgroup_pairs, and typecoord_pairs must be same")
        if any([len(tc_pair) != 2 for tc_pair in typecoord_pairs]):
            raise ValueError("Each coordinate-type (typecoord) pair must contain 2 coordinate-types")
        if any([tc_pair[0] not in coord_types for tc_pair in typecoord_pairs]) \
        or any([tc_pair[1] not in coord_types for tc_pair in typecoord_pairs]):
            raise ValueError(f"Each coordinate-type must be one of these: {coord_types}")

    def _prepare(self):
        self.results = []

    def _single_frame(self):
        num_pairs = len(self.ag_pairs)
        Distances = np.empty((0))
        for i in range(num_pairs):
            pw = self.pairwise[i]
            ag1,ag2 = self.ag_pairs[i]
            tc1,tc2 = self.tc_pairs[i]
            getcrd1 = self.get_coord[tc1]
            getcrd2 = self.get_coord[tc2]
            coords1 = getcrd1(ag1)
            coords2 = getcrd2(ag2)
            if pw==2:
                if len(coords1) != len(coords2):
                    raise ValueError(f"Number of effective coordinates from each AtomGroup must be the same.")
                distance = np.linalg.norm(coords1-coords2)
            elif pw==1:
                distance_matrix = distances.distance_array(coords1,coords2)
                distance = np.flatten(distance_matrix)
            else:
                if ag1==ag2 and tc1==tc2:
                    distance_matrix = distances.distance_array(coords1,coords2)
                    num_coord = coords1.shape[0]
                else:
                    coords = np.vstack((coords1,coords2))
                    num_coord = coords.shape[0]
                    distance_matrix = distances.distance_array(coords,coords)
                distance = distance_matrix[np.triu_indices(num_coord, k = 1)]
            Distances = np.hstack((Distances,distance))
        self.results.append(Distances)

    def _conclude(self):
        self.results = np.array(self.results)

class Dihedral:
    def __init__(self, AtomGroup, type_dihedrals):
        self.atomgroup = AtomGroup
        self.dih_type = type_dihedrals
        self.results = None
    def run(self, start=None, stop=None, step=None):
        dih_type = self.dih_type
        if type(dih_type) == type('str'):
            ag = self.atomgroup
            if dih_type == 'rama':
                dihobj = dihedrals.Ramachandran(ag)
            if dih_type == 'janin':
                dihobj = dihedrals.Janin(ag)
            dihobj.run(start, stop, step)
            dih1 = dihobj.results.angles[:,:,0]*np.pi/180.0
            dih2 = dihobj.results.angles[:,:,1]*np.pi/180.0
            dih = np.hstack((dih1,dih2))
            dih_sin = np.sin(dih)
            dih_cos = np.cos(dih)
            self.results = np.hstack((dih_sin,dih_cos))
        elif type(dih_type) == type([]):
            self.results = []
            for i in range(len(dih_type)):
                ag = self.atomgroup[i]
                dt = dih_type[i]
                if dt[0] == 'rama':
                    dihobj = dihedrals.Ramachandran(ag)
                if dt[0] == 'janin':
                    dihobj = dihedrals.Janin(ag)
                dihobj.run(start, stop, step)
                dih = dihobj.results.angles[:,:,dt[1]]#.reshape(-1,1)
                self.results.append(dih*np.pi/180.0)
            self.results = np.hstack(self.results)
        

# Feature dictionary

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

def get_FeatureDictionary(base_selstr):
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
    get_secnd_features = [Dihedral, Dihedral, \
                          Distance, Distance, Distance, Distance, Distance, Distance]
    get_arguments = [get_argdih,get_argdih,get_argdis,get_argdis,get_argdis,get_argdis,get_argdis,get_argdis]
    secnd_feature_type = [0,1] + [2]*6
    numcodes = len(selcodes)
    FeatureDict = {}
    for i in range(numcodes):
        FeatureDict[selcodes[i]] = [arg_selcodes[i], selstrs[i], get_secnd_features[i], get_arguments[i], secnd_feature_type[i]]
    return FeatureDict



# Extract all features

def get_argdih(Universe, SelStr, CoordType):
    AtomGroup = Universe.select_atoms(SelStr)
    args = [AtomGroup, CoordType]
    return args

def get_argdis(Universe, SelStr, CoordTypes):
    AtomGroups = [[Universe.select_atoms(sel) for sel in SelStr]]
    args = [AtomGroups, [CoordTypes]]
    return args

def get_FeatureValues(Universe, BaseSelstr, SelCodes, start=None, stop=None, step=None, outFile='Features.nc'):
    FeatureDict = get_FeatureDictionary(BaseSelstr)
    featureIDs = []
    FeatureVals = []
    for selcode in SelCodes:
        arg_selcode,selstr,get_secnd_features,get_arguments,secnd_feature_type = FeatureDict[selcode]
        args = get_arguments(Universe, selstr, arg_selcode)
        secnd_features = get_secnd_features(*args)
        secnd_features.run(start, stop, step)
        features = secnd_features.results
        num_features = features.shape[1]
        featureIDs += [secnd_feature_type]*num_features
        FeatureVals.append(features)
    FeatureVals = np.hstack(FeatureVals)
    featureIDs = np.array(featureIDs)
    ncFile = nc.Dataset(outFile,'w')
    ncFile.createDimension('frames',FeatureVals.shape[0])
    ncFile.createDimension('values',FeatureVals.shape[1])
    ncFeat = ncFile.createVariable("features","f4",("frames","values"))
    ncFeat[:,:] = FeatureVals
    ncFile.close()
    return FeatureVals, featureIDs
