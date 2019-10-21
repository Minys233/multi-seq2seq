from multiseq2seq.data.utils import preprocess_smiles, uniq
from joblib import Parallel, delayed
import argparse
from typing import TextIO

"""
Preprocessing ZINC15 data.
"""


def readline(fp, lines, sep=' '):
    """
    Read a chunk of lines of a file, memory efficient.
    Read the first non-blank text block of each line, as SMILES sequence.
    :param TextIO fp: Opened file object.
    :param int lines: Process how many line per time.
    :param str sep: Separation char used for `str.split()`
    :return: List of contents of lines, stripped.
    :rtype: list
    """
    lst = []
    for i in range(lines):
        line = fp.readline().strip().split(sep)[0]
        if line:
            lst.append(line)
    return lst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help="Input raw SMILES file, one per line. You may need pre-preprocess.")
    parser.add_argument('outfile', type=str, help="Save preprocessed SMILES sequences file.")
    parser.add_argument('batch', type=int, help="Lines per time, for saving your memory, 100000 -- 4-6GB mem.")
    parser.add_argument('njobs', type=int, default=-1, help="N subprocesses, default all.")
    args = parser.parse_args()
    infn, outfn, batch, njobs = args.infile, args.outfile, args.batch, args.njobs

    inf = open(infn, 'r')
    outf = open(outfn, 'w')
    idx = 0
    while True:
        smilst = readline(inf, batch)
        if not smilst: # empty list, file exausted.
            break
        print(f"{(idx * batch) // 1000}k - {((idx + 1) * batch) // 1000}k smiles")
        res = Parallel(n_jobs=njobs)(delayed(preprocess_smiles)(s) for s in smilst)
        for s in res:
            if s is None: continue
            outf.write(s + '\n')
        idx += 1

    inf.close()
    outf.close()

    uniq(outfn)

