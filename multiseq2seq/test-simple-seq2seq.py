import torch
import argparse
import yaml
import pandas as pd
from pathlib import Path
from rdkit import Chem
import random
import cloudpickle
import multiseq2seq.QSAR.QSARdata as QSAR
from multiseq2seq.model import Encoder, Decoder, Seq2Seq
from functools import partial
from joblib import Parallel, delayed
import time

parser = argparse.ArgumentParser()
parser.add_argument('CKPT', type=str, help="Checkpoint file")
parser.add_argument('CONFIG', type=str, help="Checkpoint config")
# parser.add_argument('CSVFILE', type=str, help="CSV file for testing trained model")
# parser.add_argument('SMILES_COL', type=str, help='Column name for smiles')
# parser.add_argument('LABEL_COL', type=str, help='Column name for labels')
# parser.add_argument('TYPE', type=str, choices=['classification', 'regression'], help="Task type")
parser.add_argument('--DEVICE', dest='DEVICE', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
# parser.add_argument('--FIELDS', dest='FIELDS', type=str, help='Fields for src and trg')
args = parser.parse_args()

assert Path(args.CKPT).is_file(), "CKPT file do no exist"
assert Path(args.CONFIG).is_file(), "CONFIG file do no exist"
# assert Path(args.FIELDS).is_file(), "FIELDS file do no exist"
# assert Path(args.CSVFILE).is_file(), "CSVFILE file do no exist"
# assert args.FIELDS is not None, "FIELDS could not be None"

# Network Hyper-parameters
CONFIG = yaml.load(open(args.CONFIG))
DEVICE = args.DEVICE
MODULE_NAME = CONFIG['MODULE_NAME']
BATCH_SIZE = CONFIG['BATCH_SIZE']
EMBED_DIM = CONFIG['EMBED_DIM']
HIDDEN_DIM = CONFIG['HIDDEN_DIM']
NUM_LAYERS = CONFIG['NUM_LAYERS']
ENCODER_DROPOUT = CONFIG['ENCODER_DROPOUT']
DECODE_DROPOUT = CONFIG['DECODE_DROPOUT']
BIDIRECTIONAL = CONFIG['BIDIRECTIONAL']
N_EPOCHS = CONFIG['N_EPOCHS']
COMMENT = CONFIG['COMMENT']
# Task parameters
# CSVFILE = args.CSVFILE
# SMILES_COL = args.SMILES_COL
# LABELS_COL = args.LABEL_COL
# TYPE = args.TYPE

# Others
random.seed(20191013)
RANDOM_STATE = random.getstate()
REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
# print(f"Loading Fields from {args.FIELDS}")
checkpoint = cloudpickle.load(open(args.CKPT, 'rb'))
SRC, TRG = checkpoint['SRC'], checkpoint['TRG']
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)


def encode(model, s, sos=2, eos=3, pad=1):
    s = SRC.numericalize(s).squeeze()
    arr_in = pad * torch.ones(s.shape[0] + 2 + 2, dtype=torch.long).to('cuda')
    arr_in[0], arr_in[len(s) + 1] = sos, eos
    arr_in[1:len(s) + 1] = s
    arr_in = torch.unsqueeze(arr_in, 1)
    with torch.no_grad():
        return model(arr_in)[1].flatten().cpu().numpy()


if __name__ == '__main__':
    enc = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DECODE_DROPOUT, BIDIRECTIONAL)
    dec = Decoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DECODE_DROPOUT, BIDIRECTIONAL)
    model = Seq2Seq(enc, dec, device=DEVICE).to(DEVICE)
    model.load_state_dict(checkpoint['model'])

    # Parallel(n_jobs=-1)(delayed(QSAR.evaluation_helper)(encode_fn=partial(encode, model.encoder), qsar_data=d, file='/home/minys/experiment_result.txt') for d in QSAR.classification_tasks+QSAR.regression_tasks)
    #

    for d in QSAR.classification_tasks+QSAR.regression_tasks:
        t1 = time.time()
        QSAR.evaluation_helper(encode_fn=partial(encode, model.encoder), qsar_data=d, file=f'/archive/log/result/{Path(args.CKPT).name}----{Path(args.CONFIG).name}')
        delta = time.time() - t1
        print(f"Time used: {delta} s")