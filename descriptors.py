#
# Copyright (C) 2001-2017 greg Landrum and Rational Discovery LLC
#
#   @@ All Rights Reserved @@
#  This file is part of the RDKit.
#  The contents are covered by the terms of the BSD license
#  which is included in the file license.txt, found at the root
#  of the RDKit source tree.
#
from collections import \
    abc  # this won't work in python2, but we don't support that any more

import rdkit.Chem.ChemUtils.DescriptorUtilities as _du
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.EState.EState import (MaxAbsEStateIndex, MaxEStateIndex, MinAbsEStateIndex,
                                      MinEStateIndex)
from rdkit.Chem.QED import qed

#
#  Copyright (c) 2023, RDKit Hackathon 2023, implemented by:
# - Arsenio Cruz
# - José-Manuel Gally
# - Axel Pahl
# - Vincenzo Palmacci
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   * Neither the name of Novartis Institutes for BioMedical Research Inc.
#     nor the names of its contributors may be used to endorse or promote
#     products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
Spacial score (SPS) is an empirical scoring system to express the spacial complexity of a compound
in an uniform manner and on a highly granular scale for ranking and comparison between molecules. [1]
SPS takes into account the fraction of sp3 hybridized carbons and the fraction of stereogenic carbons.

By default, this module generates the normalized spacial score (nSPS), which is a variation of the SPS score
that considers the size of the molecule.
To obtain the nSPS score the SPS score is divided by the total number of heavy atoms in the analyzed molecule.

SPS = sum(h*s*r*n*n)
nSPS = SPS/a

Where:
h = Atom hybridisation term
s = Stereoisomeric term
r = Non-aromatic ring term
n = Number of heavy atom neighbors
a = Total number of heavy atoms in the molecule

The SPS function in this module takes a mol object and returns either the absolute score (normalize=False) or the score normalized by the number of heavy atoms (normalize=True (default)).

The original code implementation can be found at: https://github.com/frog2000/Spacial-Score/blob/main/spacial_score.py

[1] Krzyzanowski, A.; Pahl, A.; Grigalunas, M.; Waldmann, H. Spacial Score─A Comprehensive Topological Indicator for Small-Molecule Complexity. J. Med. Chem. 2023. https://doi.org/10.1021/acs.jmedchem.3c00689.
"""
from rdkit import Chem

# import rdkit.Chem.Descriptors as Desc
from rdkit.Chem import rdmolops
from rdkit.Chem.ChemUtils.DescriptorUtilities import setDescriptorVersion
from collections import defaultdict


@setDescriptorVersion(version="1.0.0")
def SPS(mol, normalize=True):
  """Calculates the SpacialScore descriptor. By default, the score is normalized by the number of heavy atoms (nSPS) resulting in a float value,
    otherwise (normalize=False) the absolute score is returned as an integer.
    """
  return _SpacialScore(mol, normalize=normalize).score


class _SpacialScore:
  """Class intended for calculating spacial score (SPS) and size-normalised SPS (nSPS) for small organic molecules"""

  def __init__(self, mol, normalize=True):
    if mol is None:
      raise ValueError("No valid molecule object found.")
    molCp = Chem.Mol(mol)
    rdmolops.FindPotentialStereoBonds(molCp)
    self.mol = molCp  # mol is supposed to be a valid RDKit Mol object
    self.normalize = normalize  # if true nSPS, otherwise SPS
    self.hyb_score = {}
    self.stereo_score = {}
    self.ring_score = {}
    self.bond_score = {}
    self.chiral_idxs = self.findStereoAtomIdxs()
    self.doublebonds_stereo = self.findDoubleBondsStereo()
    # calculate SPS
    self.score = self.calculateSpacialScore()
    # return nSPS
    if normalize:
      self.score /= self.mol.GetNumHeavyAtoms()

  def findStereoAtomIdxs(self, includeUnassigned=True):
    """Finds indices of atoms that are (pseudo)stereo/chiralcentres, in respect to the attached groups (does not account for double bond isomers)"""
    stereo_centers = Chem.FindMolChiralCenters(
      self.mol,
      includeUnassigned=includeUnassigned,
      includeCIP=False,
      useLegacyImplementation=False,
    )
    stereo_idxs = [atom_idx for atom_idx, _ in stereo_centers]
    return stereo_idxs

  def findDoubleBondsStereo(self):
    """Finds indeces of stereo double bond atoms (E/Z)"""
    db_stereo = {}
    for bond in self.mol.GetBonds():
      if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
        db_stereo[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = bond.GetStereo()
    return db_stereo

  def calculateSpacialScore(self):
    """Calculates the total spacial score for a molecule"""
    score = 0
    for atom in self.mol.GetAtoms():
      atom_idx = atom.GetIdx()
      self.hyb_score[atom_idx] = self._accountForHybridisation(atom)
      self.stereo_score[atom_idx] = self._accountForStereo(atom_idx)
      self.ring_score[atom_idx] = self._accountForRing(atom)
      self.bond_score[atom_idx] = self._accountForNeighbors(atom)
      score += self._calculateScoreForAtom(atom_idx)  # absolute score
    return score

  def _calculateScoreForAtom(self, atom_idx):
    """Calculates the total score for a single atom in a molecule"""
    atom_score = (self.hyb_score[atom_idx] * self.stereo_score[atom_idx] *
                  self.ring_score[atom_idx] * self.bond_score[atom_idx])
    return atom_score

  _hybridisations = defaultdict(lambda: 4)
  _hybridisations.update({
    Chem.HybridizationType.SP: 1,
    Chem.HybridizationType.SP2: 2,
    Chem.HybridizationType.SP3: 3
  })

  def _accountForHybridisation(self, atom):
    """Calculates the hybridisation score for a single atom in a molecule"""
    return self._hybridisations[atom.GetHybridization()]

  def _accountForStereo(self, atom_idx):
    """Calculates the stereo score for a single atom in a molecule"""
    if atom_idx in self.chiral_idxs:
      return 2
    for bond_atom_idxs, stereo in self.doublebonds_stereo.items():
      if stereo != Chem.BondStereo.STEREONONE and atom_idx in bond_atom_idxs:
        return 2
    return 1

  def _accountForRing(self, atom):
    """Calculates the ring score for a single atom in a molecule"""
    if atom.GetIsAromatic():  # aromatic rings are not promoted
      return 1
    if atom.IsInRing():
      return 2
    return 1

  def _accountForNeighbors(self, atom):
    """Calculates the neighbour score for a single atom in a molecule
        The second power allows to account for branching in the molecular structure"""
    return atom.GetDegree()**2

def _isCallable(thing):
  return isinstance(thing, abc.Callable) or \
              hasattr(thing, '__call__')


_descList = []


def _setupDescriptors(namespace):
  global _descList, descList
  from rdkit.Chem import (Crippen, Descriptors3D, Fragments, GraphDescriptors, Lipinski, MolSurf,
                          )
  from rdkit.Chem.EState import EState_VSA
  _descList.clear()

  mods = [GraphDescriptors] # MolSurf, EState_VSA, Lipinski, Crippen, Fragments

  otherMods = [Chem]

  for nm, thing in tuple(namespace.items()):
    if nm[0] != '_' and _isCallable(thing):
      _descList.append((nm, thing))

  others = []
  for mod in otherMods:
    tmp = dir(mod)
    for name in tmp:
      if name[0] != '_':
        thing = getattr(mod, name)
        if _isCallable(thing):
          others.append(name)

  for mod in mods:
    tmp = dir(mod)

    for name in tmp:
      if name[0] != '_' and name[-1] != '_' and name not in others:
        # filter out python reference implementations:
        if name[:2] == 'py' and name[2:] in tmp:
          continue
        if name == 'print_function':
          continue
        thing = getattr(mod, name)
        if _isCallable(thing):
          namespace[name] = thing
          _descList.append((name, thing))

  descList = _descList


MolWt = lambda *x, **y: _rdMolDescriptors._CalcMolWt(*x, **y)
MolWt.version = _rdMolDescriptors._CalcMolWt_version
MolWt.__doc__ = """The average molecular weight of the molecule

  >>> MolWt(Chem.MolFromSmiles('CC'))
  30.07
  >>> MolWt(Chem.MolFromSmiles('[NH4+].[Cl-]'))
  53.49...

"""


def HeavyAtomMolWt(x):
  return MolWt(x, True)


HeavyAtomMolWt.__doc__ = """The average molecular weight of the molecule ignoring hydrogens

  >>> HeavyAtomMolWt(Chem.MolFromSmiles('CC'))
  24.02...
  >>> HeavyAtomMolWt(Chem.MolFromSmiles('[NH4+].[Cl-]'))
  49.46

"""
HeavyAtomMolWt.version = "1.0.0"

ExactMolWt = lambda *x, **y: _rdMolDescriptors.CalcExactMolWt(*x, **y)
ExactMolWt.version = _rdMolDescriptors._CalcExactMolWt_version
ExactMolWt.__doc__ = """The exact molecular weight of the molecule

  >>> ExactMolWt(Chem.MolFromSmiles('CC'))
  30.04...
  >>> ExactMolWt(Chem.MolFromSmiles('[13CH3]C'))
  31.05...

"""


def NumValenceElectrons(mol):
  """ The number of valence electrons the molecule has

    >>> NumValenceElectrons(Chem.MolFromSmiles('CC'))
    14
    >>> NumValenceElectrons(Chem.MolFromSmiles('C(=O)O'))
    18
    >>> NumValenceElectrons(Chem.MolFromSmiles('C(=O)[O-]'))
    18
    >>> NumValenceElectrons(Chem.MolFromSmiles('C(=O)'))
    12

    """
  tbl = Chem.GetPeriodicTable()
  return sum(
    tbl.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetFormalCharge() + atom.GetTotalNumHs()
    for atom in mol.GetAtoms())


NumValenceElectrons.version = "1.1.0"


def NumRadicalElectrons(mol):
  """ The number of radical electrons the molecule has
      (says nothing about spin state)

    >>> NumRadicalElectrons(Chem.MolFromSmiles('CC'))
    0
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[CH3]'))
    0
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[CH2]'))
    1
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[CH]'))
    2
    >>> NumRadicalElectrons(Chem.MolFromSmiles('C[C]'))
    3

    """
  return sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())


NumRadicalElectrons.version = "1.1.0"


def _ChargeDescriptors(mol, force=False):
  if not force and hasattr(mol, '_chargeDescriptors'):
    return mol._chargeDescriptors
  chgs = rdPartialCharges.ComputeGasteigerCharges(mol)
  minChg = 500.
  maxChg = -500.
  for at in mol.GetAtoms():
    chg = float(at.GetProp('_GasteigerCharge'))
    minChg = min(chg, minChg)
    maxChg = max(chg, maxChg)
  res = (minChg, maxChg)
  mol._chargeDescriptors = res
  return res


def MaxPartialCharge(mol, force=False):
  _, res = _ChargeDescriptors(mol, force)
  return res


MaxPartialCharge.version = "1.0.0"


def MinPartialCharge(mol, force=False):
  res, _ = _ChargeDescriptors(mol, force)
  return res


MinPartialCharge.version = "1.0.0"


def MaxAbsPartialCharge(mol, force=False):
  v1, v2 = _ChargeDescriptors(mol, force)
  return max(abs(v1), abs(v2))


MaxAbsPartialCharge.version = "1.0.0"


def MinAbsPartialCharge(mol, force=False):
  v1, v2 = _ChargeDescriptors(mol, force)
  return min(abs(v1), abs(v2))


MinAbsPartialCharge.version = "1.0.0"


def _FingerprintDensity(mol, func, *args, **kwargs):
  aname = f'_{func.__name__}_{args}'
  if hasattr(mol, aname):
    fp = getattr(mol, aname)
  else:
    fp = func(*((mol, ) + args), **kwargs)
    setattr(mol, aname, fp)
  if hasattr(fp, 'GetNumOnBits'):
    val = fp.GetNumOnBits()
  else:
    val = len(fp.GetNonzeroElements())
  num_heavy_atoms = mol.GetNumHeavyAtoms()
  if num_heavy_atoms == 0:
    res = 0.0
  else:
    res = float(val) / num_heavy_atoms
  return res


def _getMorganCountFingerprint(mol, radius):
  fpg = rdFingerprintGenerator.GetMorganGenerator(radius)
  return fpg.GetSparseCountFingerprint(mol)


def FpDensityMorgan1(x):
  return _FingerprintDensity(x, _getMorganCountFingerprint, 1)


def FpDensityMorgan2(x):
  return _FingerprintDensity(x, _getMorganCountFingerprint, 2)


def FpDensityMorgan3(x):
  return _FingerprintDensity(x, _getMorganCountFingerprint, 3)


_du.setDescriptorVersion('1.0.0')(FpDensityMorgan1)
_du.setDescriptorVersion('1.0.0')(FpDensityMorgan2)
_du.setDescriptorVersion('1.0.0')(FpDensityMorgan3)

if hasattr(rdMolDescriptors, 'BCUT2D'):
  names = [
    "BCUT2D_%s" % s
    for s in ('MWHI', "MWLOW", "CHGHI", "CHGLO", "LOGPHI", "LOGPLOW", "MRHI", "MRLOW")
  ]
  _du.VectorDescriptorWrapper(_rdMolDescriptors.BCUT2D, names=names, version="1.0.0",
                              namespace=locals())

_setupDescriptors(locals())

if hasattr(rdMolDescriptors, 'CalcAUTOCORR2D'):
  names = ["AUTOCORR2D_%s" % str(i + 1) for i in range(192)]
  autocorr = _du.VectorDescriptorWrapper(_rdMolDescriptors.CalcAUTOCORR2D, names=names,
                                         version="1.0.0", namespace=locals())

  def setupAUTOCorrDescriptors():
    """Adds AUTOCORR descriptors to the default descriptor lists"""
    _setupDescriptors(namespace=autocorr.namespace)


class PropertyFunctor(rdMolDescriptors.PythonPropertyFunctor):
  """Creates a python based property function that can be added to the
    global property list.  To use, subclass this class and override the
    __call__ method.  Then create an instance and add it to the
    registry.  The __call__ method should return a numeric value.

    Example:

      class NumAtoms(Descriptors.PropertyFunctor):
        def __init__(self):
          Descriptors.PropertyFunctor.__init__(self, "NumAtoms", "1.0.0")
        def __call__(self, mol):
          return mol.GetNumAtoms()

      numAtoms = NumAtoms()
      rdMolDescriptors.Properties.RegisterProperty(numAtoms)
    """

  def __init__(self, name, version):
    rdMolDescriptors.PythonPropertyFunctor.__init__(self, self, name, version)

  def __call__(self, mol):
    raise NotImplementedError("Please implement the __call__ method")


def CalcMolDescriptors(mol, missingVal=None, silent=True):
  ''' calculate the full set of descriptors for a molecule
    
    Parameters
    ----------
    mol : RDKit molecule
    missingVal : float, optional
                 This will be used if a particular descriptor cannot be calculated
    silent : bool, optional
             if True then exception messages from descriptors will be displayed

    Returns
    -------
    dict 
         A dictionary with decriptor names as keys and the descriptor values as values
    '''
  res = {}
  for nm, fn in _descList:
    # some of the descriptor fucntions can throw errors if they fail, catch those here:
    try:
      val = fn(mol)
    except:
      if not silent:
        import traceback
        traceback.print_exc()
      val = missingVal
    res[nm] = val
  return res


# ------------------------------------
#
#  doctest boilerplate
#
def _runDoctests(verbose=None):  # pragma: nocover
  import doctest
  import sys
  failed, _ = doctest.testmod(optionflags=doctest.ELLIPSIS, verbose=verbose)
  sys.exit(failed)


if __name__ == '__main__':  # pragma: nocover
  _runDoctests()