import hashlib
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem import Descriptors
import os

"""
Functions that can be used to preprocess SMILES sequnces in the form used in the publication.
This file is modified from https://github.com/jrwnter/cddd/blob/master/cddd/preprocessing.py
So that the training data are same. Please refer to this repo and their paper for further information.

Modified: minor RDKit function calls, docstring types.
"""

REMOVER = SaltRemover.SaltRemover()
ORGANIC_ATOM_SET = {5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
#                   B, C, N, O, F, P,  S,  Cl, Br, I


def md5_hash(s):
    """
    Calculate MD5 hash of input string, for removing duplicates.
    :param str s: Input string.
    :return: Hex MD5 hash, in `bytes` type.
    :rtype: bytes
    """
    hash_object = hashlib.md5(s.encode())
    return hash_object.digest()


def canonical_smile(sml):
    """
    Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce
    :param str sml: A SMILES sequence.
    :return: A canonical SMILES sequence.
    :rtype: str
    """
    # modified here to first transform `sml` to a `Mol` instance.
    return Chem.MolToSmiles(Chem.MolFromSmiles(sml), canonical=True)


def keep_largest_fragment(sml):
    """
    Function that returns the SMILES sequence of the largest fragment for a input
    SMILES sequence.
    :param str sml: A SMILES sequence.
    :return: The canonical SMILES sequence of the largest fragment.
    :rtype: str
    """
    mol_frags = Chem.GetMolFrags(Chem.MolFromSmiles(sml), asMols=True)
    largest_mol = None
    largest_mol_size = 0
    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size
    return Chem.MolToSmiles(largest_mol)


def remove_salt_stereo(sml, remover):
    """
    Function that strips salts and removes stereochemistry information from a SMILES. :param str sml: A SMILES sequence.
    :param SaltRemover remover: RDKit's SaltRemover object.
    :return: The canonical SMILES sequence without
    salts and stereochemistry information. If any error on processing, return None instead.
    :rtype: Union[str, NoneType]
    """
    try:
        sml = Chem.MolToSmiles(remover.StripMol(Chem.MolFromSmiles(sml),
                               dontRemoveEverything=True),
                               isomericSmiles=False)
        if "." in sml:
            sml = keep_largest_fragment(sml)
    except:
        sml = None
    return sml


def organic_filter(sml):
    """
    Function that filters for organic molecules.
    :param str sml: A SMILES sequence.
    :return: If `sml` can be interpreted by RDKit and is organic.
    :rtype: bool
    """
    try:
        m = Chem.MolFromSmiles(sml)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return True
        else:
            return False
    except:
        return False


def filter_smiles(sml):
    """
    Return the canonical SMILES sequence when fulfilled rules:
        - -5 < LogP < 7
        - 12 < mol_weight < 600
        - 3 < num_heavy_atoms < 50
        - is organic (in `ORGANIC_ATOM_SET`)
    Return None Any input SMILES violate the rules will make this
    :param str sml: A SMILES sequence.
    :return: A canonical SMILES sequence or None.
    :rtype: Union[str, NoneType]
    """
    try:
        m = Chem.MolFromSmiles(sml)
        logp = Descriptors.MolLogP(m)
        mol_weight = Descriptors.MolWt(m)
        num_heavy_atoms = Descriptors.HeavyAtomCount(m)
        atom_num_list = [atom.GetAtomicNum() for atom in m.GetAtoms()]
        is_organic = set(atom_num_list) <= ORGANIC_ATOM_SET
        if ((logp > -5) & (logp < 7) &
                (mol_weight > 12) & (mol_weight < 600) &
                (num_heavy_atoms > 3) & (num_heavy_atoms < 50) &
                is_organic):
            return Chem.MolToSmiles(m)
        else:
            return None
    except:
        return None


def preprocess_smiles(sml):
    """
    Function that preprocesses a SMILES string such that it is in the same format as
    the translation model was trained on. It removes salts and stereochemistry from the
    SMILES sequnce. If the sequnce correspond to an inorganic molecule or cannot be
    interpreted by RDKit, None is returned.
    :param str sml: A SMILES sequence.
    :return: A canonical SMILES sequence or None.
    :rtype: Union[str, NoneType]
    """
    new_sml = remove_salt_stereo(sml, REMOVER)
    new_sml = filter_smiles(new_sml)
    return new_sml


def uniq(outfn):
    """
    Using GNU tool (`os.system()`) to sort and remove duplicated lines.
    :param Path outfn: The file path, `pathlib.Path` and `str` are.
    :return: Do not return, aka. returns None.
    """
    cmd_uniq = f"sort -d {str(outfn)} | uniq > {str(outfn) + '.tmp'}"
    cmd_mv = f"mv {str(outfn) + '.tmp'} {str(outfn)}"
    print("System call:", cmd_uniq)
    os.system(cmd_uniq)
    print("System call:", cmd_mv)
    os.system(cmd_mv)
