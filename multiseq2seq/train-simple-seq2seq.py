import re
import time
from pathlib import Path
import tqdm
import argparse
import torch
import random
import cloudpickle
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset

from multiseq2seq.utils import init_weight, count_parameters, save_model, save_config
from multiseq2seq.dataset import FileSpliter
from multiseq2seq.model import Encoder, Decoder, Seq2Seq

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('from_train', type=str, help="Translation source")
parser.add_argument('to_train', type=str, help="Translation target")
parser.add_argument('from_valid', type=str, help="Translation source")
parser.add_argument('to_valid', type=str, help="Translation target")
parser.add_argument('--MODULE_NAME', dest='MODULE_NAME', type=str, default="foobar")
parser.add_argument('--DEVICE', dest='DEVICE', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', type=int, default=1024)
parser.add_argument('--EMBED_DIM', dest='EMBED_DIM', type=int, default=40)
parser.add_argument('--HIDDEN_DIM', dest='HIDDEN_DIM', type=int, default=256)
parser.add_argument('--NUM_LAYERS', dest='NUM_LAYERS', type=int, default=1)
parser.add_argument('--ENCODER_DROPOUT', dest='ENCODER_DROPOUT', type=float, default=0.0)
parser.add_argument('--DECODER_DROPOUT', dest='DECODER_DROPOUT', type=float, default=0.0)
parser.add_argument('--BIDIRECTIONAL', dest='BIDIRECTIONAL', type=bool, default=True)
parser.add_argument('--N_EPOCHS', dest='N_EPOCHS', type=int, default=100)
parser.add_argument('--COMMENT', dest='COMMENT', type=str, default='Foo Bar')
parser.add_argument('--CKPT_BASE', dest='CKPT_BASE', type=str, default='.', help="Location to save checkpoints")
parser.add_argument('--TB_DIR', dest='TB_DIR', type=str, default='./logs', help="Tensorboard log dir")
parser.add_argument('--FIELDS', dest='FIELDS', type=str, default=None, help='Fields for src and trg')
args = parser.parse_args()

# Network Hyper-parameters
MODULE_NAME = args.MODULE_NAME
DEVICE = args.DEVICE
BATCH_SIZE = args.BATCH_SIZE
EMBED_DIM = args.EMBED_DIM
HIDDEN_DIM = args.HIDDEN_DIM
NUM_LAYERS = args.NUM_LAYERS
ENCODER_DROPOUT = args.ENCODER_DROPOUT
DECODE_DROPOUT = args.DECODER_DROPOUT
BIDIRECTIONAL = args.BIDIRECTIONAL
N_EPOCHS = args.N_EPOCHS

CONFIG = {
    'MODULE_NAME': MODULE_NAME,
    'BATCH_SIZE': BATCH_SIZE,
    'EMBED_DIM': EMBED_DIM,
    'HIDDEN_DIM': HIDDEN_DIM,
    'NUM_LAYERS': NUM_LAYERS,
    'ENCODER_DROPOUT': ENCODER_DROPOUT,
    'DECODE_DROPOUT': DECODE_DROPOUT,
    'BIDIRECTIONAL': BIDIRECTIONAL,
    'N_EPOCHS': N_EPOCHS,
    'COMMENT': args.COMMENT
}


# Data Preprocess
random.seed(20191013)
RANDOM_STATE = random.getstate()
REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'
if args.FIELDS is not None and Path(args.FIELDS).is_file():
    print(f"Loading Fields from {args.FIELDS}")
    SRC, TRG = cloudpickle.load(open(args.FIELDS, 'rb'))
else:
    SRC = Field(tokenize=lambda smi: re.findall(REGEX_SML, smi),
                init_token='<sos>',
                eos_token='<eos>',
                lower=False)

    TRG = Field(tokenize=lambda smi: re.findall(REGEX_SML, smi)[::-1],
                init_token='<sos>',
                eos_token='<eos>',
                lower=False)

# Logger
WRITER = SummaryWriter(log_dir=args.TB_DIR)
CKPT_BASE = Path(args.CKPT_BASE)


# Training
def train(model, iterator, optimizer, criterion, epoch):
    model.train()
    epoch_loss, epoch_accu = 0, 0
    pbar = tqdm.tqdm(iterator, unit='Batch', desc="Training")
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        output = model(batch.src, batch.trg)
        # skip <eos>
        output = output[1:].view(-1, output.shape[-1])
        trg = batch.trg[1:].view(-1)
        # output: [(seq_len-1)*batch_size, output_dim]
        # trg: [(seq_len-1)*batch_size]
        accu = torch.mean((torch.argmax(output, dim=1) == trg).type(torch.float))
        loss = criterion(output, trg)
        WRITER.add_scalar('Train/Loss', loss.item(), epoch*len(iterator)+i)
        WRITER.add_scalar('Train/Accuracy', accu.item(), epoch * len(iterator) + i)
        loss.backward()
        # clip the weight
        # nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.3f} accu: {accu.item()*100:.3f}%")
        epoch_loss += loss.item()
        epoch_accu += accu.item()
        del loss
        del accu
        del output
    epoch_loss, epoch_accu = epoch_loss / len(iterator), epoch_accu / len(iterator)
    WRITER.add_scalar('Train/Epoch-Loss', epoch_loss, epoch)
    WRITER.add_scalar('Train/Epoch-Accu', epoch_accu, epoch)
    return epoch_loss, epoch_accu


def evaluate(model: Seq2Seq, iterator, criterion, epoch):
    model.eval()
    epoch_loss, epoch_accu = 0, 0
    pbar = tqdm.tqdm(iterator, unit='Batch', desc="Evaluating")
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            output = model(batch.src, batch.trg, 0)
            # output: [trg_len, batch_size, output_dim=vocab_dim]
            output = output[1:].view(-1, output.shape[-1])
            trg = batch.trg[1:].view(-1)
            # loss
            loss = criterion(output, trg)
            accu = torch.mean((output.argmax(dim=1) == trg).type(torch.float))
            WRITER.add_scalar('Validate/Loss', loss.item(), epoch * len(iterator) + i)
            WRITER.add_scalar('Validate/Accuracy', accu.item(), epoch * len(iterator) + i)
            epoch_loss += loss.item()
            epoch_accu += accu.item()
            del loss
            del accu
    epoch_loss, epoch_accu = epoch_loss / len(iterator), epoch_accu / len(iterator)
    WRITER.add_scalar('Validate/Epoch-Loss', epoch_loss, epoch)
    WRITER.add_scalar('Validate/Epoch-Accu', epoch_accu, epoch)
    return epoch_loss, epoch_accu


if __name__ == "__main__":
    # Data preparation
    train_split = FileSpliter(args.from_train, args.to_train, split=10, name='train')
    valid_split = FileSpliter(args.from_valid, args.to_valid, split=5, name='valid')

    # build vocabulary
    if not hasattr(SRC, 'vocab'):
        SRC.build_vocab(open(args.from_train))
    if not hasattr(TRG, 'vocab'):
        TRG.build_vocab(open(args.to_train))
    print(SRC.vocab.stoi)
    print(TRG.vocab.stoi)
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

    # Instantiate models
    enc = Encoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DECODE_DROPOUT, BIDIRECTIONAL)
    dec = Decoder(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DECODE_DROPOUT, BIDIRECTIONAL)

    model = Seq2Seq(enc, dec, device=DEVICE).to(DEVICE)
    model.apply(init_weight)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])
    # start training
    getext = lambda x: '.'+x.name.rsplit('.', 1)[-1]
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        for idx, (from_train_fn, to_train_fn) in enumerate(train_split):
            # for every bulk of training data
            # 1. train
            print(f'On {from_train_fn} - {to_train_fn}')
            train_iterator = TranslationDataset(path=str(train_split.location / train_split.name),
                                                exts=(getext(from_train_fn), getext(to_train_fn)),
                                                fields=[('src', SRC), ('trg', TRG)])
            train_iterator = BucketIterator(train_iterator, batch_size=BATCH_SIZE, device=DEVICE, train=True,
                                            shuffle=True)
            train(model, train_iterator, optimizer, criterion, epoch)
            del train_iterator
            # 2. validation
            for vidx, (from_valid_fn, to_valid_fn) in enumerate(valid_split):
                valid_iterator = TranslationDataset(path=str(valid_split.location / valid_split.name),
                                                    exts=(getext(from_valid_fn), getext(to_valid_fn)),
                                                    fields=[('src', SRC), ('trg', TRG)])
                valid_iterator = BucketIterator(valid_iterator, batch_size=BATCH_SIZE, device=DEVICE, train=False,
                                                shuffle=True)
                evaluate(model, valid_iterator, criterion, epoch)
                del valid_iterator
                valid_split.shuffle()
                break
            # 3. save
            save_model(MODULE_NAME, model,
                       epoch, optimizer, SRC, TRG, CKPT_BASE, f"block{idx}",
                       comment=args.COMMENT)
            save_config(CONFIG, CKPT_BASE / (MODULE_NAME + '.config'))

        deltamin, deltasec = divmod(time.time() - start_time, 60)
        print(f"\nEpoch: {epoch + 1:3d} | Time: {deltamin}min {deltasec:.1f}s")

WRITER.close()
