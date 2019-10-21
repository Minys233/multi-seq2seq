# shellcheck disable=SC2155
export PYTHONPATH=`readlink -f ..`
EMBED_DIM=40
HIDDEN_DIM=128
BATCH_SIZE=512
DEVICE='cuda:0'
LOG_BASE=/home/minys/log
DATA_BASE=/home/minys/data/preprocessed


NUM_LAYERS=1
MODULE_NAME="seq2seq-randomIO-${HIDDEN_DIM}-${NUM_LAYERS}"
python train-random-input-output.py \
    --MODULE_NAME "${MODULE_NAME}" --DEVICE "${DEVICE}"\
    --BATCH_SIZE "${BATCH_SIZE}" --EMBED_DIM "${EMBED_DIM}"\
    --HIDDEN_DIM "${HIDDEN_DIM}" --NUM_LAYERS "${NUM_LAYERS}" \
    --ENCODER_DROPOUT 0.3 --DECODER_DROPOUT 0.3\
    --BIDIRECTIONAL 0 --N_EPOCHS 3\
    --COMMENT "random-smiles-input-output"\
    --CKPT_BASE "${LOG_BASE}"/checkpoints\
    --TB_DIR "${LOG_BASE}/tfboards/${MODULE_NAME}"\
    --FIELDS /home/minys/Projects/multi-seq2seq/Fields.pkl\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi


NUM_LAYERS=2
MODULE_NAME="seq2seq-randomIO-${HIDDEN_DIM}-${NUM_LAYERS}"
python train-random-input-output.py \
    --MODULE_NAME "${MODULE_NAME}" --DEVICE "${DEVICE}"\
    --BATCH_SIZE "${BATCH_SIZE}" --EMBED_DIM "${EMBED_DIM}"\
    --HIDDEN_DIM "${HIDDEN_DIM}" --NUM_LAYERS "${NUM_LAYERS}" \
    --ENCODER_DROPOUT 0.3 --DECODER_DROPOUT 0.3\
    --BIDIRECTIONAL 0 --N_EPOCHS 3\
    --COMMENT "random-smiles-input-output"\
    --CKPT_BASE "${LOG_BASE}"/checkpoints\
    --TB_DIR "${LOG_BASE}/tfboards/${MODULE_NAME}"\
    --FIELDS /home/minys/Projects/multi-seq2seq/Fields.pkl\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi


NUM_LAYERS=4
MODULE_NAME="seq2seq-randomIO-${HIDDEN_DIM}-${NUM_LAYERS}"
python train-random-input-output.py \
    --MODULE_NAME "${MODULE_NAME}" --DEVICE "${DEVICE}"\
    --BATCH_SIZE "${BATCH_SIZE}" --EMBED_DIM "${EMBED_DIM}"\
    --HIDDEN_DIM "${HIDDEN_DIM}" --NUM_LAYERS "${NUM_LAYERS}" \
    --ENCODER_DROPOUT 0.3 --DECODER_DROPOUT 0.3\
    --BIDIRECTIONAL 0 --N_EPOCHS 3\
    --COMMENT "random-smiles-input-output"\
    --CKPT_BASE "${LOG_BASE}"/checkpoints\
    --TB_DIR "${LOG_BASE}/tfboards/${MODULE_NAME}"\
    --FIELDS /home/minys/Projects/multi-seq2seq/Fields.pkl\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.train.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi\
    "${DATA_BASE}"/pubchem-zinc12.valid.smi
