import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
import pandas as pd
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
import numpy as np

if tf.test.is_gpu_available():
    LSTM = keras.layers.CuDNNLSTM
else:
    LSTM = keras.layers.LSTM

SEED = 20190926
OUTF = open('saspr_baseline.txt', 'w')
print(f"Method\tDataset\tMetrics\tMean(CV)\tStd(CV)", file=OUTF)

def mol2vec_features(mol2vec_path, dataframe, smiles_col, target_col, pad_to):
    model = word2vec.Word2Vec.load(mol2vec_path)
    # validate smiles first!
    smiles_lst = dataframe[smiles_col].to_numpy()
    labels_lst = dataframe[target_col].to_numpy()
    idx = []
    for i, s in enumerate(smiles_lst):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
        except Exception as e:
            continue
        idx.append(i)
    smiles_lst = smiles_lst[np.array(idx)]
    labels_lst = labels_lst[np.array(idx)]
    # mol2vec embeddings
    mollst = [Chem.MolFromSmiles(x) for x in smiles_lst]
    sentences = [mol2alt_sentence(x, 1) for x in mollst]
    features = np.zeros([len(mollst), pad_to, model.vector_size])
    labels = np.array(labels_lst)
    print("mean: ", labels.mean(), "std: ", labels.std())
    for idx, sentence in enumerate(sentences):
        count = 0
        for word in sentence:
            if count == pad_to:
                break
            try:
                features[idx, count] = model.wv[word]
                count += 1
            except KeyError as e:
                pass
    assert features.shape[0] == labels.shape[0]
    return features, labels


# refer to https://github.com/thushv89/attention_keras
# and documentation of tf.keras.layers.Layer
# for how to define a custom layer.
class SelfAttentiveLayer(keras.layers.Layer):
    def __init__(self, da, r, **kwargs):
        super(SelfAttentiveLayer, self).__init__(**kwargs)
        self.da = da
        self.r = r

    def build(self, input_shape):
        # concat of outputs of BiLSTM, with shape (batch_size, n, vector_size)
        assert isinstance(input_shape, tf.TensorShape)
        n, vector_size = input_shape[1], input_shape[2]
        self.W1 = self.add_weight(name='W1_attention',
                                  shape=tf.TensorShape((self.da, vector_size)),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2_attention',
                                  shape=tf.TensorShape((self.r, self.da)),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttentiveLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # ignore kwargs inorder to keep same signature with meta class
        # the input should have already been concatenated!
        if isinstance(inputs, list):
            raise TypeError('Argument `inputs` should have already been concatenated. '
                            'Do not directly pass the output of LSTM or GRU to this layer, '
                            'use `tf.concat` first to merge them into single tensor.'
                            'If your input sequences have variable length, please padd first.')
        # TODO: add mask for padded sequence
        H = inputs
        # input : (batch_size, n, vector_size)
        # 1. transpose to (batch_size, vector_size, n)
        H_trans = tf.transpose(inputs, perm=[0, 2, 1])
        # 2. matmul with W1: (self.da, vector_size) * (batch_size, vector_size, n) = (batch_size, self.da, n)
        # 3. tanh activation
        after_w1 = tf.tanh(self.W1 @ H_trans)  # matmul
        # 4. matmul with W2: (self.r, self.da) * (batch_size, self.da, n) = (batch_size, self.r, n)
        # 5. softmax activation
        # 6. the output matrix A will has a shape of (batch_size, self.r, n)
        A = tf.nn.softmax(self.W2 @ after_w1, axis=1)  # maybe wrong! which axis???
        # 7. matmul A with input H to get output Ma:
        # (batch_size, self.r, n) * (batch_size, n, vector_size) = (batch_size, self.r, vector_size)
        Ma = A @ H
        # NO BIAS
        return Ma


# def build_sa_bilstm_model(pad_to, vector_size, lstm_hidden, da, r):
#     inputs = keras.layers.Input(shape=(pad_to, vector_size), dtype='float32')
#     sequence = keras.layers.Bidirectional(LSTM(lstm_hidden, return_sequences=True))(inputs)
#     # print(sequence)
#     # concatenate = keras.layers.Concatenate()(*sequence)
#     self_attention = SelfAttentiveLayer(da, r)(sequence)
#     flatten = keras.layers.Flatten()(self_attention)
#     y = keras.layers.Dense(1)(flatten)
#     # may be activation could be added here
#     model = keras.models.Model(inputs=inputs, outputs=y)
#     return model


def baseline_model(pad_to, vector_size, lstm_hidden, da, r, **config):
    if 'type' not in config:
        raise ValueError("Include `type` in **config parameter!")
    # if 'tail' not in config:
    #     raise TypeError("Please include `tail` in config (a list of integers) to specify network architecture.")
    if config['type'] not in ['classification', 'regression']:
        raise ValueError("config['type'] must in ['classification', 'regression']")

    if config['type'] == 'classification':
        if 'num_class' not in config:
            raise ValueError("num_class is required config for classification")
        classes = config['num_class']
        inputs = keras.layers.Input(shape=(pad_to, vector_size), dtype='float32')
        sequence = keras.layers.Bidirectional(LSTM(lstm_hidden, return_sequences=True))(inputs)
        self_attention = SelfAttentiveLayer(da, r)(sequence)
        flatten = keras.layers.Flatten()(self_attention)
        for i in config['tail']:
            flatten = keras.layers.Dense(i, activation='relu')(flatten)
        y = keras.layers.Dense(classes, activation='softmax', use_bias=False)(flatten)
        model = keras.models.Model(inputs=inputs, outputs=y)
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[keras.metrics.SparseCategoricalAccuracy()]
                      )
        return model
    elif config['type'] == 'regression':
        inputs = keras.layers.Input(shape=(pad_to, vector_size), dtype='float32')
        sequence = keras.layers.Bidirectional(LSTM(lstm_hidden, return_sequences=True))(inputs)
        self_attention = SelfAttentiveLayer(da, r)(sequence)
        flatten = keras.layers.Flatten()(self_attention)
        for i in config['tail']:
            flatten = keras.layers.Dense(i)(flatten)
        y = keras.layers.Dense(1)(flatten)
        model = keras.models.Model(inputs=inputs, outputs=y)
        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.MSE,
                      metrics=[keras.metrics.MSE]
                      )
        return model


def evaluate(csv_path, smi_col, label_col, mol2vec_path, pad_to=70, lstm_hidden=300, da=100, r=20, **config):
    df = pd.read_csv(csv_path)
    features, labels = mol2vec_features(mol2vec_path, df, smi_col, label_col, pad_to)
    features, labels = features.astype('single'), labels.astype('single')
    _, _, vector_size = features.shape
    # TODO: cross-validation, not random shuffle

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    # features_vali, features_test, labels_vali, labels_test = \
    #     train_test_split(features, labels, test_size=0.5)
    model = baseline_model(pad_to, vector_size, lstm_hidden, da, r, **config)
    model.summary()
    # mol2vec can be raw, since it has been normalized.
    # normalize labels when regression
    if config['type'] == 'regression':
        m, s = labels_train.mean(), labels_train.std()
        labels_train = (labels_train - m) / s
        labels_test = (labels_test - m) / s
        # labels_vali = (labels_vali - m) / s

    print(len(labels_train), "training samples")
    print(len(labels_test), "validation & testing samples")
    train_dataset = Dataset.from_tensor_slices((features_train, labels_train)).shuffle(buffer_size=256).batch(32, drop_remainder=True)

    earlystop_callback = keras.callbacks.EarlyStopping(patience=3)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'/data/checkpoints/saspr-{config["name"]}-{pad_to}-{lstm_hidden}-{da}-{r}.ckpt',
        save_best_only=True)
    model.fit(train_dataset,
              validation_data=(features_test, labels_test),
              epochs=100,
              callbacks=[earlystop_callback, checkpoint_callback]
              )

    if config['type'] == 'classification':
        from sklearn.metrics import roc_auc_score, accuracy_score
        predict = np.array(model.predict(features_test), dtype=np.float)
        truth = np.array(labels_test, dtype=np.int)
        predict_score = []
        for t, p in zip(truth, predict):
            predict_score.append(p[int(t)])
        predict_score = np.array(predict_score)
        print(f"Class.\t{config['name']}\tROC-AUC\t{roc_auc_score(truth, predict_score):.3f}", file=OUTF)
        print(f"Class.\t{config['name']}\taccu.\t{accuracy_score(truth, predict.argmax(axis=1)):.3f}", file=OUTF)

    if config['type'] == 'regression':
        from scipy.stats import pearsonr
        predict = np.array(model.predict(features_test)).ravel() * s + m
        truth = np.array(labels_test).ravel() * s + m
        MSE = ((predict - truth) ** 2).mean()
        r2 = pearsonr(truth, predict)[0]**2
        print(f"Regres.\t{config['name']}\tMSE\t{MSE:.3f}", file=OUTF)
        print(f"Regres.\t{config['name']}\tR^2\t{r2:.3f}", file=OUTF)


if __name__ == "__main__":
    config_ames = {
        'name': 'ames',
        'type': 'classification',
        'tail': [],
        'num_class': 2
    }
    evaluate(csv_path='/home/minys/Projects/multi-seq2seq/multiseq2seq/QSAR/classification/ames/ames.csv',
             smi_col='smiles',
             label_col='Activity',
             mol2vec_path='/data/data/pretrained_models/mol2vec_model_300dim.pkl',
             pad_to=70,
             lstm_hidden=150,
             da=20,
             r=15,
             **config_ames)
    keras.backend.clear_session()

    config_bace = {
        'name': 'bace',
        'type': 'classification',
        'tail': [],
        'num_class': 2
    }
    evaluate(csv_path='/home/minys/Projects/multi-seq2seq/multiseq2seq/QSAR/classification/bace/bace.csv',
             smi_col='mol',
             label_col='Class',
             mol2vec_path='/data/data/pretrained_models/mol2vec_model_300dim.pkl',
             pad_to=70,
             lstm_hidden=150,
             da=20,
             r=15,
             **config_bace)
    keras.backend.clear_session()


    config_bbbp = {
        'name': 'bbbp',
        'type': 'classification',
        'tail': [],
        'num_class': 2
    }
    evaluate(csv_path='/home/minys/Projects/multi-seq2seq/multiseq2seq/QSAR/classification/bbbp/BBBP.csv',
             smi_col='smiles',
             label_col='p_np',
             mol2vec_path='/data/data/pretrained_models/mol2vec_model_300dim.pkl',
             pad_to=70,
             lstm_hidden=150,
             da=20,
             r=15,
             **config_bbbp)
    keras.backend.clear_session()


    config_esol = {
        'name': 'esol',
        'type': 'regression',
        'tail': [10],
    }
    evaluate(csv_path='/home/minys/Projects/multi-seq2seq/multiseq2seq/QSAR/regression/ESOL/delaney-processed.csv',
             smi_col='smiles',
             label_col='measured log solubility in mols per litre',
             mol2vec_path='/data/data/pretrained_models/mol2vec_model_300dim.pkl',
             pad_to=70,
             lstm_hidden=150,
             da=20,
             r=15,
             **config_esol)
    keras.backend.clear_session()


    config_free = {
        'name': 'FreeSolv',
        'type': 'regression',
        'tail': [10],
    }
    evaluate(csv_path='/home/minys/Projects/multi-seq2seq/multiseq2seq/QSAR/regression/FreeSolv/SAMPL.csv',
             smi_col='smiles',
             label_col='expt',
             mol2vec_path='/data/data/pretrained_models/mol2vec_model_300dim.pkl',
             pad_to=70,
             lstm_hidden=150,
             da=20,
             r=15,
             **config_free)
    keras.backend.clear_session()


    config_lipo = {
        'name': 'lipo',
        'type': 'regression',
        'tail': [10],
    }
    evaluate(csv_path='/home/minys/Projects/multi-seq2seq/multiseq2seq/QSAR/regression/lipophilicity/Lipophilicity.csv',
             smi_col='smiles',
             label_col='exp',
             mol2vec_path='/data/data/pretrained_models/mol2vec_model_300dim.pkl',
             pad_to=70,
             lstm_hidden=150,
             da=20,
             r=15,
             **config_lipo)
    keras.backend.clear_session()

OUTF.close()
