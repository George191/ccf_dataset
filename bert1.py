import codecs

import pandas as pd
import codecs, gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score


train_lines = codecs.open('data/Second_DataSet.csv', encoding='utf-8').readlines()[1:]
train_df = pd.DataFrame({
    'id': [x[:32] for x in train_lines],
    'ocr': [x[33:].strip() for x in train_lines]
})
train_label = pd.read_csv('data/Second_DataSet_Label.csv', encoding='utf-8')
train_df = pd.merge(train_df, train_label, on='id')

test_lines = codecs.open('data/Second_TestDataSet.csv', encoding='utf-8').readlines()[1:]
test_df = pd.DataFrame({
    'id': [x[:32] for x in test_lines],
    'ocr': [x[33:].strip() for x in test_lines]
})

print(train_df.info())
print(test_df.info())

batch_size = 64
maxlen = 510
config_path = 'errnie_model/bert_config.json'
checkpoint_path = 'errnie_model/bert_model.ckpt'
dict_path = 'errnie_model/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=64, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []





def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


class F1_Metrics(Callback):

    def __init__(self, val_data, batch_size = 32):
        super().__init__()
        self.validation_data = val_data
        self.validation_data_generator = val_data.__iter__()
        self.batch_size = batch_size
        self.metric = 0
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
     if not('val_f1-score' in self.params['metrics']):
        self.params['metrics'].append('val_f1-score')

    def on_epoch_end(self, epoch, logs={}):
     #print(self.validation_data[0][0], "validation data 0", self.validation_data[1], "validation data 1")
     batches = self.validation_data.__len__()
     print("batches ", batches)
     total = len(self.validation_data.data)#batches * self.batch_size
     print("total ", total)
   
     val_predict = np.zeros((total))
     val_targ = np.zeros((total))
     
     #self.validation_data_generator = self.validation_data.__iter__()   
     for batch in list(range(batches-1)):
         #print(batch,"\n",  self.batch_size,"batch size")
         xVal, yVal = next(self.validation_data_generator)
         #print(yVal,yVal.shape)
         #exit()
         val_predict[batch * self.batch_size : (batch+1) * self.batch_size] = np.argmax(np.asarray(self.model.predict(xVal)), axis=1)#.round()
         val_targ[batch * self.batch_size : (batch+1) * self.batch_size] = np.argmax(np.asarray(yVal), axis=1)
         #print(np.argmax(np.asarray(yVal), axis=1))
       
     if batches == 1:
         batch = 0
     else:
         batch = batch + 1
     xVal, yVal = next(self.validation_data_generator)
     val_predict[batch * self.batch_size:] = np.argmax(np.asarray(self.model.predict(xVal)), axis=1)
     val_targ[batch * self.batch_size:] = np.argmax(yVal, axis=1)
 
     val_predict = np.squeeze(val_predict)

     print(val_predict,'val predict', val_targ,'val_targ')
     _val_f1 = f1_score(val_targ, val_predict, average='macro')
     logs['val_f1-score'] = _val_f1
     if _val_f1 > self.metric:
         self.metric = _val_f1

     _val_recall = recall_score(val_targ, val_predict,average='macro')
     _val_precision = precision_score(val_targ, val_predict, average='macro')
     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)
     
     print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
     return 
     

DATA_LIST = []
for data_row in train_df.iloc[:].itertuples():
    DATA_LIST.append((data_row.ocr, to_categorical(data_row.label, 3)))
DATA_LIST = np.array(DATA_LIST)

DATA_LIST_TEST = []
for data_row in test_df.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.ocr, to_categorical(0, 3)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)
val_f1_list = list()

x_data = train_df['ocr']
y_data = train_df['label']

def run_cv(nfold, data, data_label, data_test):
    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(x_data, y_data)
    train_model_pred = np.zeros((len(data), 3))
    test_model_pred = np.zeros((len(data_test), 3))

    for i, (train_fold, test_fold) in enumerate(kf):
        print(train_fold)
        print(test_fold)
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]

        model = build_bert(3)
        early_stopping = EarlyStopping(monitor='val_f1-score', patience=3)
        plateau = ReduceLROnPlateau(monitor="val_f1-score", verbose=1, mode='max', factor=0.5, patience=2)
        checkpoint = ModelCheckpoint('./bert_dump2/' + str(i) + '.hdf5', monitor='val_f1-score',
                                     verbose=2, save_best_only=True, mode='max', save_weights_only=True)

        train_D = data_generator(X_train, shuffle=True, batch_size=batch_size)
        valid_D = data_generator(X_valid, shuffle=True, batch_size=batch_size)
        test_D = data_generator(data_test, shuffle=False, batch_size=batch_size)
        
        f1_metrics = F1_Metrics(valid_D, batch_size = batch_size)
        
        print(f'KF_{i}_f1-socre: {f1_metrics}')
        
        val_f1_list.append(f1_metrics.metric)
        
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=50,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint, f1_metrics],
        )

        # model.load_weights('./bert_dump/' + str(i) + '.hdf5')

        # return model
        train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)

        del model
        gc.collect()
        K.clear_session()

        # break

    return train_model_pred, test_model_pred


train_model_pred, test_model_pred = run_cv(5, DATA_LIST, None, DATA_LIST_TEST)



train_prob = train_model_pred.tolist()
train_df['label'] = train_prob
train_df[['id', 'label']].to_csv('Train_pred_baseline_prob.csv', index=None)

train_pred = [np.argmax(x) for x in train_model_pred]
train_df['label'] = train_pred
train_df[['id', 'label']].to_csv('Train_pred_baseline.csv', index=None)

test_model_pred = test_model_pred / nfold
test_prob = test_model_pred.tolist()
test_pred = [np.argmax(x) for x in test_model_pred]

test_df['label'] = test_pred

test_df[['id', 'label']].to_csv('Test_pred_baseline.csv', index=None)

test_df['label'] = test_prob

test_df[['id', 'label']].to_csv('Test_pred_baseline_prob.csv', index=None)

val_f1 = np.mean(val_f1_list)
val_f1_list.append(val_f1)
val_f1_series = pd.DataFrame(val_f1_list).T

val_f1_series.columns = list(range(nfold)) + ['mean val f1']
val_f1_series.to_csv('Val_f1.csv', index=None)
with open('performance_filename.txt','a',encoding='utf-8') as p:
   p.writelines('Val_f1_.csv' + '\n')
   p.writelines(str(val_f1) + '\n\n')
