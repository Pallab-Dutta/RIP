import os
from pymol import cmd
import numpy as np
import pandas as pd

idx1 = None
idx2 = None
name1 = None
name2 = None

def normalize(X):
    normX = X/np.sum(X)
    return normX

def scaled_diff(X1,X2):
    Xd = X1-X2
    sXd = -1 + (Xd-np.min(Xd))/(np.max(Xd)-np.min(Xd))*2
    return sXd

def minmaxScale(X):
    mmsX = (X-np.min(X))/(np.max(X)-np.min(X))
    return mmsX

def get_imp_diff(Xd, cutoff=0.3):
    aXd = minmaxScale(abs(Xd))
    argsort_aXd = np.argsort(aXd)[::-1]
    sorted_aXd = np.sort(aXd)[::-1]
    cumsum_aXd = np.cumsum(sorted_aXd)
    sum_aXd = np.sum(sorted_aXd)
    partsum_aXd = cumsum_aXd/sum_aXd
    partsum_aXd = partsum_aXd <= cutoff
    last_true_idx = np.sum(partsum_aXd)
    cutoff_idx1 = argsort_aXd[:last_true_idx]
    cutoff_idx2 = argsort_aXd[last_true_idx:]
    return cutoff_idx1, cutoff_idx2

def get_alnstr(alnFile):
    rf = open(alnFile,'r')
    adata = rf.read()
    rf.close()
    adata = adata.split('\n\n')
    aln_list = [[],[],[]]

    for e in adata:
        for i,ee in enumerate(e.split('\n')):
            aln_list[i].append(ee[13:])

    alnstr = []
    for e in aln_list:
        estr = ''.join(e)
        alnstr.append(estr)

    return alnstr

def get_aln_index(seq,alnstr):
    IDX = [True]*len(alnstr)
    j=0
    for i,e in enumerate(alnstr):
        if e==' ':
            if seq[i]=='-' or seq[i]==' ':
                IDX.pop(i-j)
                j+=1
            else:
                IDX[i-j]=False
    return IDX


@cmd.extend
def RIPsuper(mol1, mol2, object=False):
    global name1, name2, idx1, idx2
    name1 = cmd.get_object_list(mol1)[0]
    name2 = cmd.get_object_list(mol2)[0]
    if object:
        alnobj = object
    else:
        alnobj = f'aln_{name1}_{name2}'
    cmd.super(mol1, mol2, object=alnobj)
    cmd.hide('cgo', alnobj)
    cmd.save(f'{name1}_{name2}.aln', alnobj)
    seq1,seq2,alnstr = get_alnstr(f'{name1}_{name2}.aln')
    os.remove(f'{name1}_{name2}.aln')
    idx1 = get_aln_index(seq1,alnstr)
    idx2 = get_aln_index(seq2,alnstr)

@cmd.extend
def RIPalign(mol1, mol2, object=False):
    global name1, name2, idx1, idx2
    name1 = cmd.get_object_list(mol1)[0]
    name2 = cmd.get_object_list(mol2)[0]
    if object:
        alnobj = object
    else:
        alnobj = f'aln_{name1}_{name2}'
    cmd.align(mol1, mol2, object=alnobj)
    cmd.hide('cgo', alnobj)
    cmd.save(f'{name1}_{name2}.aln', alnobj)
    seq1,seq2,alnstr = get_alnstr(f'{name1}_{name2}.aln')
    os.remove(f'{name1}_{name2}.aln')
    idx1 = get_aln_index(seq1,alnstr)
    idx2 = get_aln_index(seq2,alnstr)

@cmd.extend
def RIPcealign(mol1, mol2, object=False):
    global name1, name2, idx1, idx2
    name1 = cmd.get_object_list(mol1)[0]
    name2 = cmd.get_object_list(mol2)[0]
    if object:
        alnobj = object
    else:
        alnobj = f'aln_{name1}_{name2}'
    cmd.cealign(mol1, mol2, object=alnobj)
    cmd.hide('cgo', alnobj)
    cmd.save(f'{name1}_{name2}.aln', alnobj)
    seq1,seq2,alnstr = get_alnstr(f'{name1}_{name2}.aln')
    os.remove(f'{name1}_{name2}.aln')
    idx1 = get_aln_index(seq1,alnstr)
    idx2 = get_aln_index(seq2,alnstr)

def get_delRIP(RIP1, RIP2):
    """
    Calculates the difference between Residue Importance Projections (RIP) of two different kinases.
    """
    RIP1 = np.array(RIP1)
    RIP2 = np.array(RIP2)
    delRIP = minmaxScale(normalize(RIP1))-minmaxScale(normalize(RIP2))
    return delRIP

@cmd.extend
def RIPcolor(obj=None, cmap='rainbow'):
    cmd.alter(name1, "b=0")
    cmd.alter(name2, "b=0")

    if isinstance(obj, list):
        resids1, resids2, deltaRIP = obj
        for resid,value in zip(resids1,deltaRIP):
            cmd.alter(f"{name1} and resi {int(resid)}", f"b={value}")
        for resid,value in zip(resids2,deltaRIP):
            cmd.alter(f"{name2} and resi {int(resid)}", f"b={value}")
        
        cmd.spectrum("b",cmap,f"{name1}")
        cmd.spectrum("b",cmap,f"{name2}")

        if cmap=='rainbow':
            cmd.ramp_new("delRIP_cbar", name1, [-1, 1], cmap)
            cmd.wizard("message", f"-1: {name2}, 1: {name1}")
        else:
            cmd.delete("delRIP_cbar")
            cmd.wizard("message", f"{cmap} - undefined pymol colorbar")
        
        cmd.recolor()

    else:
        try:
            name = cmd.get_object_list(obj)[0]
            allRIP = np.loadtxt(f'{name}.rip')
            resids, values = allRIP.T
            
            cmd.hide('spheres', name)

            for resid,value in zip(resids,values):
                cmd.alter(f"{name} and resi {int(resid)}", f"b={value}")

            cmd.spectrum("b",cmap,f"{name}")

            if cmap=='rainbow':
                cmd.ramp_new(f"{name}_cbar", name, [0, 1], cmap)
                cmd.wizard("message", "0: least important, 1: most important")
            else:
                cmd.delete(f"{name}_cbar")
                cmd.wizard("message", f"{cmap} - undefined pymol colorbar")
            
            cmd.recolor()
        except TypeError:
            cmd.wizard("message", "Please provide a molecular object")

@cmd.extend
def delRIP(threshold, color=True, cmap='rainbow'):
    allRIP1 = np.loadtxt(f'{name1}.rip')
    allRIP2 = np.loadtxt(f'{name2}.rip')
    RIP1 = allRIP1[idx1,:]
    RIP2 = allRIP2[idx2,:]
    deltaRIP = get_delRIP(RIP1[:,1], RIP2[:,1])
    highlight_indices, nonhighlight_indices = get_imp_diff(deltaRIP,cutoff=float(threshold)/100)
    resids1 = RIP1[:,0]
    resids2 = RIP2[:,0]

    cmd.hide('spheres',name1)
    cmd.hide('spheres',name2)
    
    for hi in highlight_indices:
        cmd.show('spheres','%s and resi %d and n. N'%(name1, int(resids1[hi])))
        cmd.show('spheres','%s and resi %d and n. N'%(name2, int(resids2[hi])))
    
    #for nh in nonhighlight_indices:    
    #    cmd.hide('spheres','%s and resi %d and n. N'%(name1, int(resids1[nh])))
    #    cmd.hide('spheres','%s and resi %d and n. N'%(name2, int(resids2[nh])))
    
    if color:
        delRIP_obj = [resids1, resids2, deltaRIP]
        RIPcolor(delRIP_obj, cmap)
