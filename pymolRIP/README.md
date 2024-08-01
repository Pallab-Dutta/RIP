# Pymol RIP visualizer

## Introduction
- Project the Residue Importance as colors on the selected molecule
- Compute the $\Delta RIP$ for two aligned molecules

## Project Residue Importance
### Usage
```pymol
RIPcolor obj, cmap
```
### Required Arguments
- obj = any object selection (within one single object though)

### Optional Arguments
- cmap = any pymol colormap (default is rainbow which is also available for pymol colorbar)

### Example
```pymol
PyMOL> load ABL.pdb
PyMOL> RIPcolor ABL
```

## Project $\Delta$ Residue Importance
### Usage
```pymol
RIPsuper/RIPalign/RIPcealign mol1, mol2, object
delRIP threshold, cmap
```
### Required Arguments
- mol1 = string: atom selection of mobile object
- mol2 = string: atom selection of target object
- threshold = top (threshold)% of important residues contributing in $\Delta RIP$

### Optional Arguments
- object = string: name of alignment object to create (default: aln_mol1_mol2)
- cmap = any pymol colormap (default is rainbow which is also available for pymol colorbar)

### Example
```pymol
PyMOL> load ABL.pdb, ABL
PyMOL> load PKA.pdb, PKA
PyMOL> RIPsuper ABL, PKA
PyMOL> delRIP 30
```
