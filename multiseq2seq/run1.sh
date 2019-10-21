# shellcheck disable=SC2155
export PYTHONPATH=`readlink -f ..`
EMBED_DIM=40
HIDDEN_DIM=512
BATCH_SIZE=512
LOG_BASE=/archive/log #/home/minys/log
DATA_BASE=/data/data/preprocessed #/home/minys/data/preprocessed


NUM_LAYERS=1
python train-simple-seq2seq.py \
    --MODULE_NAME "seq2seq-${HIDDEN_DIM}-${NUM_LAYERS}" --DEVICE 'cuda:0'\
    --BATCH_SIZE "${BATCH_SIZE}" --EMBED_DIM "${EMBED_DIM}"\
    --HIDDEN_DIM "${HIDDEN_DIM}" --NUM_LAYERS "${NUM_LAYERS}" \
    --ENCODER_DROPOUT 0.3 --DECODER_DROPOUT 0.3\
    --BIDIRECTIONAL 0 --N_EPOCHS 3\
    --COMMENT "Faster"\
    --CKPT_BASE "${LOG_BASE}"/checkpoints\
    --TB_DIR "${LOG_BASE}/tfboards/seq2seq-${HIDDEN_DIM}-${NUM_LAYERS}"\
    --FIELDS /home/minys/Projects/multi-seq2seq/Fields.pkl\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi


NUM_LAYERS=2
python train-simple-seq2seq.py \
    --MODULE_NAME "seq2seq-${HIDDEN_DIM}-${NUM_LAYERS}" --DEVICE 'cuda:0'\
    --BATCH_SIZE "${BATCH_SIZE}" --EMBED_DIM "${EMBED_DIM}"\
    --HIDDEN_DIM "${HIDDEN_DIM}" --NUM_LAYERS "${NUM_LAYERS}" \
    --ENCODER_DROPOUT 0.3 --DECODER_DROPOUT 0.3\
    --BIDIRECTIONAL 0 --N_EPOCHS 3\
    --COMMENT "Faster"\
    --CKPT_BASE "${LOG_BASE}"/checkpoints\
    --TB_DIR "${LOG_BASE}/tfboards/seq2seq-${HIDDEN_DIM}-${NUM_LAYERS}"\
    --FIELDS /home/minys/Projects/multi-seq2seq/Fields.pkl\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi


NUM_LAYERS=4
python train-simple-seq2seq.py \
    --MODULE_NAME "seq2seq-${HIDDEN_DIM}-${NUM_LAYERS}" --DEVICE 'cuda:0'\
    --BATCH_SIZE "${BATCH_SIZE}" --EMBED_DIM "${EMBED_DIM}"\
    --HIDDEN_DIM "${HIDDEN_DIM}" --NUM_LAYERS "${NUM_LAYERS}" \
    --ENCODER_DROPOUT 0.3 --DECODER_DROPOUT 0.3\
    --BIDIRECTIONAL 0 --N_EPOCHS 3\
    --COMMENT "Faster"\
    --CKPT_BASE "${LOG_BASE}"/checkpoints\
    --TB_DIR "${LOG_BASE}/tfboards/seq2seq-${HIDDEN_DIM}-${NUM_LAYERS}"\
    --FIELDS /home/minys/Projects/multi-seq2seq/Fields.pkl\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi
