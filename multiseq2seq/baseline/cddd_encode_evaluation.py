from pathlib import Path
import subprocess
import argparse
import os
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

"""
Simple wrapper for cddd, to extract discriptors.
Commandline arguments are managed by cddd function.
NOTE: Change to python environment with tensorflow=1.10.0 installed!
NOTE: Try to run the cddd code with :
      python run_cddd.py --input xxx.csv \
                         --smiles_header xxxx \
                         --preprocess \
                         --model_dir /path/xxx \
                         --use_gpu \
                         --device 0 \
                         --cpu_threads 20
"""

parser = argparse.ArgumentParser()
parser.add_argument('run_cddd_file',
                    help="Path to run_cddd.py, same at https://github.com/jrwnter/cddd/blob/master/cddd/run_cddd.py")
parser.add_argument('model_dir',
                    help="Pretrained cddd model, download this "
                         "with https://github.com/jrwnter/cddd/blob/master/download_default_model.sh")
parser.add_argument('python',
                    help="Python executable which contains all dependencies as https://github.com/jrwnter/cddd")
parser.add_argument('threads',
                    help="CPU threads")
parser.add_argument('gpu_device',
                    help="GPU device for tensorflow")

qsardir = Path(os.path.realpath(__file__)).parent.parent / 'QSAR'


def cddd_encode(args):
    cmd = f'{args.python} {args.run_cddd_file} ' \
          '--preprocess ' \
          '--use_gpu ' \
          f'--device {args.gpu_device} ' \
          f'--cpu_threads {args.threads} ' \
          f'--model_dir {args.model_dir} ' \
          '--input %s ' \
          '--output %s ' \
          '--smiles_header %s '
    # classification
    # 1. bace
    print("1. bace")
    bacedir = qsardir / 'classification' / 'bace'
    bace_cmd = cmd % (str(bacedir / 'bace.csv'),
                      str(bacedir / 'bace-cddd.csv'),
                      'mol')
    subprocess.run(bace_cmd, shell=True, check=True)

    # 2. bbbp
    bbbpdir = qsardir / 'classification' / 'bbbp'
    bbbp_cmd = cmd % (str(bbbpdir / 'BBBP.csv'),
                      str(bbbpdir / 'BBBP-cddd.csv'),
                      'smiles')
    subprocess.run(bbbp_cmd, shell=True, check=True)

    # 3. pcba
    pcbadir = qsardir / 'classification' / 'pcba'
    pcba_cmd = cmd % (str(pcbadir / 'pcba.csv'),
                      str(pcbadir / 'pcba-cddd.csv'),
                      'smiles')
    subprocess.run(pcba_cmd, shell=True, check=True)

    # 4. ames
    amesdir = qsardir / 'classification' / 'ames'
    ames_cmd = cmd % (str(amesdir / 'ames.csv'),
                      str(amesdir / 'ames-cddd.csv'),
                      'smiles')
    subprocess.run(ames_cmd, shell=True, check=True)

    # regression
    # 1. ESOL
    esoldir = qsardir / 'regression' / 'ESOL'
    esol_cmd = cmd % (str(esoldir / 'delaney-processed.csv'),
                      str(esoldir / 'delaney-processed-cddd.csv'),
                      'smiles')
    subprocess.run(esol_cmd, shell=True, check=True)

    # 2. lipophilicity
    lipodir = qsardir / 'regression' / 'lipophilicity'
    lipo_cmd = cmd % (str(lipodir / 'Lipophilicity.csv'),
                      str(lipodir / 'Lipophilicity-cddd.csv'),
                      'smiles')
    subprocess.run(lipo_cmd, shell=True, check=True)

    # 3. FreeSolv
    freesolvdir = qsardir / 'regression' / 'FreeSolv'
    freesolv_cmd = cmd % (str(freesolvdir / 'SAMPL.csv'),
                          str(freesolvdir / 'SAMPL-cddd.csv'),
                          'smiles')
    subprocess.run(freesolv_cmd, shell=True, check=True)

    print('Done!')


def load_cddd_csv(fpath, label_col):
    df = pd.read_csv(fpath).dropna()
    emb_cols = [f"cddd_{i+1}" for i in range(512)]
    smiles = df['new_smiles'].to_numpy()
    valid_idx = df.iloc[:, 0].to_numpy()
    embeddings = df.loc[:, emb_cols].to_numpy()
    labels = df[label_col].to_numpy()
    return smiles, valid_idx, embeddings, labels


if __name__ == "__main__":
    # encode
    flist = [
        qsardir / 'classification' / 'bace' / 'bace-cddd.csv',
        qsardir / 'classification' / 'bbbp' / 'BBBP-cddd.csv',
        qsardir / 'classification' / 'pcba' / 'pcba-cddd.csv',
        qsardir / 'classification' / 'ames' / 'ames-cddd.csv',
        qsardir / 'regression' / 'ESOL' / 'delaney-processed-cddd.csv',
        qsardir / 'regression' / 'lipophilicity' / 'Lipophilicity-cddd.csv',
        qsardir / 'regression' / 'FreeSolv' / 'SAMPL-cddd.csv'
    ]
    is_encoded = True
    for p in flist:
        if not p.is_file():
            is_encoded = False
    if not is_encoded:
        args = parser.parse_args()
        cddd_encode(args)
    # evaluation
    # classification
    # bace
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[0], label_col='Class')
    print(smiles.shape, valid_idx.shape, embeddings.shape, labels.shape)
    # bbbp
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[1], label_col='p_np')
    print(smiles.shape, valid_idx.shape, embeddings.shape, labels.shape)
    # pcba
    # TODO: figure out how to use pcba data, there are too many labels!
    # ames
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[3], label_col='Activity')
    print(smiles.shape, valid_idx.shape, embeddings.shape, labels.shape)

    # regression
    # ESOL
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[4],
                                                          label_col='ESOL predicted log solubility in mols per litre')
    print(smiles.shape, valid_idx.shape, embeddings.shape, labels.shape)
    # lipophilicity
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[5], label_col='exp')
    print(smiles.shape, valid_idx.shape, embeddings.shape, labels.shape)
    # FreeSolv
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[6], label_col='expt')
    print(smiles.shape, valid_idx.shape, embeddings.shape, labels.shape)



