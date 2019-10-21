from pathlib import Path
import subprocess
import argparse
import os
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

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


def load_cddd_csv(fpath, label_col, shuffle=True):
    df = pd.read_csv(fpath).dropna()
    emb_cols = [f"cddd_{i+1}" for i in range(512)]
    smiles = df['new_smiles'].to_numpy()
    valid_idx = df.iloc[:, 0].to_numpy()
    embeddings = df.loc[:, emb_cols].to_numpy()
    labels = df[label_col].to_numpy()
    assert len(smiles) == len(valid_idx) == len(embeddings) == len(labels)
    if shuffle:
        idx = np.random.permutation(len(valid_idx))
        smiles = smiles[idx]
        valid_idx = valid_idx[idx]
        embeddings = embeddings[idx]
        labels = labels[idx]
    return smiles, valid_idx, embeddings, labels


def _check_normalize(features, labels):
    assert features.shape[0] == labels.shape[0]
    normalize = StandardScaler()
    normalize.fit(features)
    features = normalize.transform(features)
    return features, labels


def evaluation(clf, features, labels, scorefn=None, cv=5):
    """
    Evaluate the cddd fingerprints with support vector machines.
    :param Union[SVC, SVR] clf: estimator object implementing ‘fit’, according to sklearn's estimator object.
    :param np.ndarray features: The data to fit, shape (n_samples, n_features)
    :param np.ndarray labels: The label corresponds to `features`, shape (n_samples)
    :param scorefn: a score function `fn(estimator, x, y) -> score`, higher, the better
    :param int cv: Cross-validation split, default 5.
    :return: Raw scores of evalutaion, for each cross-validation.
    """
    features, labels = _check_normalize(features, labels)
    scores = cross_val_score(clf, features, labels, cv=cv, scoring=scorefn)
    return scores


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
    print(f"Method\tDataset\tMetrics\tMean(CV)\tStd(CV)")
    # classification

    def auc_score(e, x, y):
        p = e.predict_proba(x)[:, 1]
        return roc_auc_score(y, p)

    def print_class_result(dataset, auc_roc, acc):
        print(f"Class.\t{dataset}\tAUC-ROC\t{np.mean(auc_roc):.3f}\t{np.std(auc_roc):.3f}")
        print(f"Class.\t{dataset}\taccu.\t{np.mean(acc):.3f}\t{np.std(acc):.3f}")
    # bace
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[0], label_col='Class')
    auc_roc = evaluation(SVC(probability=True, gamma='scale'), embeddings, labels, scorefn=auc_score)
    acc = evaluation(SVC(probability=True, gamma='scale'), embeddings, labels, scorefn=None)
    print_class_result('bace', auc_roc, acc)

    # bbbp
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[1], label_col='p_np')
    auc_roc = evaluation(SVC(probability=True, gamma='scale'), embeddings, labels, scorefn=auc_score)
    acc = evaluation(SVC(probability=True, gamma='scale'), embeddings, labels, scorefn=None)
    print_class_result('bbbp', auc_roc, acc)

    # pcba
    # TODO: figure out how to use pcba data, there are too many labels!

    # ames
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[3], label_col='Activity')
    auc_roc = evaluation(SVC(probability=True, gamma='scale'), embeddings, labels, scorefn=auc_score)
    acc = evaluation(SVC(probability=True, gamma='scale'), embeddings, labels, scorefn=None)
    print_class_result('ames', auc_roc, acc)

    # ===========================
    # regression
    def mse_score(e, x, y):
        p = e.predict(x)
        return np.mean((p-y)**2)

    def print_regress_result(dataset, mse, r2):
        print(f"Regres.\t{dataset}\tMSE\t{np.mean(mse):.3f}\t{np.std(mse):.3f}")
        print(f"Regres.\t{dataset}\tR^2\t{np.mean(r2):.3f}\t{np.std(r2):.3f}")

    # ESOL
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[4],
                                                          label_col='ESOL predicted log solubility in mols per litre')
    mse = evaluation(SVR(gamma='scale'), embeddings, labels, scorefn=mse_score)
    r2 = evaluation(SVR(gamma='scale'), embeddings, labels, scorefn=None)
    print_regress_result("ESOL", mse, r2)

    # lipophilicity
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[5], label_col='exp')
    mse = evaluation(SVR(gamma='scale'), embeddings, labels, scorefn=mse_score)
    r2 = evaluation(SVR(gamma='scale'), embeddings, labels, scorefn=None)
    print_regress_result("lipo", mse, r2)

    # FreeSolv
    smiles, valid_idx, embeddings, labels = load_cddd_csv(flist[6], label_col='expt')
    mse = evaluation(SVR(gamma='scale'), embeddings, labels, scorefn=mse_score)
    r2 = evaluation(SVR(gamma='scale'), embeddings, labels, scorefn=None)
    print_regress_result("FreeSolv", mse, r2)



