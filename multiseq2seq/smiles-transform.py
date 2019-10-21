from rdkit import Chem
import argparse
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
parser.add_argument('in_file')
parser.add_argument('out_file')
parser.add_argument('func', help="Python lambda function, numpy(np) may be used.")


def transform(smi, func):
    """
    Transform one SMILES.
    :param str smi: A SMILES string
    :param Callable func: A function accepts a list of SMILES and return one SMILES.
    :return: Transformed SMILES
    """
    smilst = []
    mol = Chem.MolFromSmiles(smi)
    for i in range(mol.GetNumAtoms()):
        smilst.append(Chem.MolToSmiles(mol, rootedAtAtom=i, canonical=False))
    return func(smilst)


def parenthesis_distance(smilst):
    stack, pairs, dists = [], [], []
    for s in smilst:
        for i, c in enumerate(s):
            if c == '(':
                stack.append(i)
            elif c == ')':
                pairs.append((stack.pop(), i))
        d = 0
        for x, y in pairs:
            d += abs(y - x)
        dists.append(d)
    return smilst[np.argmin(dists)]


if __name__ == '__main__':
    args = parser.parse_args()
    func = eval(args.func)
    with open(args.in_file, 'r') as infile, open(args.out_file, 'w') as outfile:
        lines = filter(lambda x: len(x)>1, infile.read().split('\n'))
        res = Parallel(n_jobs=-1)(delayed(transform)(l, func) for l in lines)
        outfile.write('\n'.join(res))

# Benchmarks on 50k SMILES
# 1. Pick the shortest one  --  want minimal branches
#     85.9% SMILES are identical
#     99.4% SMILES size shrink
#
# 2. Pick fewest () one  -- want minimal branches
#     85.9% SMILES are identical (same as above)
#     99.4% SMILES size shrink
#
# 3.

