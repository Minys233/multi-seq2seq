import gzip
import argparse

from pathlib import Path
import tqdm
from joblib import Parallel, delayed

from data.utils import preprocess_smiles, uniq


"""
Preprocess PubChem dataset (sdf.gz files).
"""


def preprocess_sdf(sdf_path):
    """
    Preprocess all molecules from a `*.sdf` file or `*.sdf.gz` file. 
    Specialized for PubChem sdf data, SMILES are directly from text file, do not fit to other sdf files.
    Note: items in retuened list may be None for invalid molecules.
    :param Path sdf_path: The pathlib.Path object for `.sdf` or `.sdf.gz` file from PubChem.
    :return: A list containing preprocessed SMILES. 
    :rtype: list
    """
    if sdf_path.name.endswith('.gz'):
        infile = gzip.open(sdf_path, 'rt')
    else:
        infile = open(sdf_path, 'r')
    
    lines = infile.readlines()
    smi_list = []
    for idx, l in enumerate(lines):
        if l == '> <PUBCHEM_OPENEYE_CAN_SMILES>\n':
            smi_list.append(lines[idx+1].strip())
    return list(map(preprocess_smiles, smi_list))


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=Path, help="Input dir containing .sdf or .sdf.gz files")
    parser.add_argument('outfile', type=Path, help="Save preprocessed SMILES sequences file.")
    parser.add_argument('njobs', type=int, default=-1, help="N subprocesses, default all.")
    args = parser.parse_args()
    indir, outfn, njobs = args.indir, args.outfile, args.njobs
    if njobs <= 0:
        raise ValueError("Please input positive `njobs` parameter.")
    
    inf_lst = list(indir.glob('**/*.sdf')) + list(indir.glob('**/*.sdf.gz'))
    # redirect RDKit's logging to stderr safely.
    progress = tqdm.tqdm(total=len(inf_lst), file=sys.stdout)
    fout = open(outfn, 'w')

    for idx in range(len(inf_lst)//njobs+1):
        batch = inf_lst[idx*njobs:(idx+1)*njobs]
        res = Parallel(n_jobs=njobs)(delayed(preprocess_sdf)(f) for f in batch)
        for res_batch in res:
            for smi in res_batch:
                if smi is not None:
                    fout.write(smi+'\n')
        progress.update(len(batch))
    
    fout.close()
    uniq(outfn)
