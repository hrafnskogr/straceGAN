import time, sys, os

#### TF SILENCER
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import io

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
    except RuntimeError as e:
        print(e)

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional, Lambda, Concatenate
from tensorflow.keras.layers import BatchNormalization, Embedding
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.keras import utils

_args = None
_ds_root = "" #".\\dataset\\"
_ds = ""            # dataset path
_ds_name = ""       # dataset name
_ds_meta = ""       # dataset metadata path

_model_save = ""
_gen_save = ""

_dir = "/"


class straceGan():
    def __init__(self, seqlen, dataset, txt2tok, tok2txt, svocab, mode, stride=10):
        self.seqlen = seqlen                # length of a sequence, will influence LSTMs
        self.Dloss = []                     # discriminator loss listt, for plotting mainly
        self.Gloss = []                     # generator loss list, for plotting mainly
        self.dataset = dataset              # list of sample composing the dataset
        self.txt2toks = txt2tok                  # token to text dictionary
        self.toks2txts = tok2txt                # text to token dictionary
        self.svocab = svocab                # vocab_size / vocabulary size / number of known tokens
        self.mode = mode
        self.stride = stride

        if(self.mode == "noise"):
            self.G = self.noise_Generator()
        elif(self.mode == "seq"):
            self.G = self.seq_Generator(stride)

        self.D = self.Discriminator()
        self.D.summary()

        self.D.trainable = False

        gan_output = self.D(self.G.output)

        self.gan = Model(self.G.input, gan_output, name="GAN")
        opt = Adam(lr=0.0002, beta_1=0.5)
        #opt = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt)
        self.gan.summary()

    def noise_Generator(self):
        gen_input = Input(shape=(self.seqlen, 1), name="Noise_in")

        generator = Dense(512)(gen_input)
        generator = Dropout(rate=0.4)(generator)
        generator = LeakyReLU(alpha=0.2)(generator)

        generator = Dense(1024)(generator)                               #1024
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)

        generator = LSTM(512, activation='tanh', return_sequences=True)(generator)
        generator = Dropout(rate=0.4)(generator)

        generator = LSTM(512, activation='tanh')(generator)

        generator = Dense(self.seqlen, activation = 'tanh')(generator)  # tanh to match the data shape []-1,1]
        gen_out = Reshape((self.seqlen, 1))(generator)

        model = Model(gen_input, gen_out, name='G')
        return model

    def seq_Generator(self, gen_seq_len):
        gen_input = Input(shape=(self.seqlen, 1), name="Sequence_in")

        generator = Dense(512)(gen_input)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)

        generator = Dense(1024)(generator)                               #1024 to be modified to this value
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)

        generator = LSTM(512, activation='tanh', return_sequences=True)(generator)

        generator = LSTM(512, activation='tanh')(generator)

        generator = Dense(gen_seq_len, activation = 'tanh')(generator)  # tanh to fit discriminator input
        gen_output = Reshape((gen_seq_len, 1))(generator)

        # Slice input from [INPUT_LEN] to [INPUT_LEN - GEN_OUTPUT]
        new_seq = Lambda(lambda x: x[:,gen_seq_len:,:], output_shape=(self.seqlen - gen_seq_len, 1))(gen_input)
        # Concatenate (sliced)input and generated output to match desired sequence length
        new_seq = Concatenate(axis=1)([new_seq, gen_output])

        model = Model(gen_input, new_seq, name='G') #gen_output, name='G')
        return model

    def Discriminator(self):
        d_in = Input(shape=(self.seqlen,1), name="Sequence_in")
        disc = LSTM(512, activation='tanh', return_sequences=True)(d_in)
        disc = Dropout(rate=0.5)(disc)
        disc = LSTM(512, activation='tanh')(disc)
        disc = Dense(512)(disc)
        disc = Dropout(rate=0.5)(disc)
        disc = LeakyReLU(alpha=0.2)(disc)
        disc = Dense(256)(disc)
        disc = LeakyReLU(alpha=0.2)(disc)
        d_out = Dense(1, activation="sigmoid")(disc)

        model = Model(d_in, d_out, name="Discriminator")
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def get_batch(self, dataset, idx, svocab, seqlen, batch_size, rand=False):
        global _args
        x_data,y_data = [], []
        tokens = []

        with open(dataset[idx], 'r') as sample:
            tokens = list(map(int, sample.read().split()))      # convert imported string of file to ints

        #if(_args.seq):
        #    start_pad = [0] * seqlen
        #    tokens = start_pad + tokens

        max =  batch_size if rand is False else (len(tokens)-seqlen)

        for i in range(0, max, 1):                       # build sequences
            try:
                x_data.append(tokens[i:i+seqlen])
                y_data.append(tokens[i+seqlen])
            except IndexError:
                print(dataset[idx])
                print(f"i+seqlen:{i+seqlen} i:{i} seqlen:{seqlen}")
                print(f"len of tokens: {len(tokens)}")
                print(f"batch_size: {batch_size}")

        if(rand):
            try:
                randidx = np.random.randint(0, len(x_data), batch_size)
                x_data = [x_data[idx] for idx in randidx]
            except:
                return False, None, None

        X = np.reshape(x_data, (batch_size, seqlen, 1))
        #X = normalize(X, svocab)
        X = neg_normalize(X, svocab)

        y = utils.to_categorical(y_data, num_classes=svocab)

        return True, X,y                                              # X: sequences y: what comes next

    def load_full_dataset(self, dataset):
        target_length = self.seqlen
        x_data, fulldata = [], []

        for data in dataset:
            with open(data, 'r') as sample:
                d = list(map(int, sample.read().split()))   # removed slicing to max seqlen, to see
                #if(len(d) == target_length):
                fulldata += d

        for i in range(0, len(fulldata) - target_length):
            x_data.append(fulldata[i:i+seqlen])

        np.random.shuffle(x_data)

        return x_data

    def get_fullseq_batch(self, dataset, batch_size, svocab):
        batch_idx = np.random.randint(0, len(dataset), batch_size)

        x_data = [dataset[idx] for idx in batch_idx]
        X = np.reshape(x_data, (batch_size, self.seqlen, 1))
        #X = normalize(X, svocab)
        X = neg_normalize(X, svocab)

        return X

    def train(self, epochs, interval, batch_size, full_seq_training=False, e=None):
        global _args
        global _dir

        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        print("\n" + 65*"=" + "\n" + 27*" " + "Work begins" + 28*" " + "\n" + 65*"=" + "\n")

        self.plot_models()

        if(full_seq_training):
            self.full_seq_data = self.load_full_dataset(self.dataset)

        if(_args.tb_dir != "NONE"):
            logdir = format_path(_args.tb_dir)
            tb_gan = tf.keras.callbacks.TensorBoard(log_dir=f"{logdir}{_args.name}{_dir}{self.mode}{_dir}gan", histogram_freq=0, write_graph=True)
            tb_gan.set_model(self.gan)

            tb_dr = tf.keras.callbacks.TensorBoard(log_dir=f"{logdir}{_args.name}{_dir}{self.mode}{_dir}d_real", histogram_freq=0, write_graph=True)
            tb_dr.set_model(self.D)

            tb_df = tf.keras.callbacks.TensorBoard(log_dir=f"{logdir}{_args.name}{_dir}{self.mode}{_dir}d_fake", histogram_freq=0, write_graph=True)
            tb_df.set_model(self.D)

        skipD = 0

        start = 0 if e is None else e

        for epoch in range(start, epochs):
            epoch_start = time.time()
            random_sample = np.random.randint(0, len(self.dataset), 1)[0]

            # block mod
            if(not full_seq_training):
                s, X,_ = self.get_batch(self.dataset, random_sample, self.svocab, self.seqlen, batch_size, True)

                while(not s):
                    random_sample = np.random.randint(0, len(self.dataset), 1)[0]
                    s, X,_ = self.get_batch(self.dataset, random_sample, self.svocab, self.seqlen, batch_size, True)
            else:
                X = self.get_fullseq_batch(self.full_seq_data, batch_size, self.svocab)


            noise = np.random.rand(len(X), self.seqlen, 1)

            if(self.mode == "noise"):
                X_hat = self.G.predict(noise)
            elif(self.mode == "seq"):
                X_hat = self.G.predict(X)

            # Train the discriminator
            if(skipD > 0):
                skipD -= 1
            else:
                d_loss_real = self.D.train_on_batch(X, real)
                d_loss_fake = self.D.train_on_batch(X_hat, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                #if(d_loss[0] < 0.45 or d_loss_fake[1] > 0.65):
                #    skipD = 2

            train = None
            if(self.mode == "noise"):
                train = noise
            elif(self.mode == "seq"):
                train = X_hat

            # train the generator
            g_loss = self.gan.train_on_batch(train, real)

            if(_args.tb_dir != "NONE"):
                tb_gan.on_epoch_end(epoch, self.named_logs(self.gan, [g_loss]))
                tb_dr.on_epoch_end(epoch, self.named_logs(self.D, d_loss_real))
                tb_df.on_epoch_end(epoch, self.named_logs(self.D, d_loss_fake))

            if(epoch % interval == 0):
                if(epoch % 50 == 0):
                    if(self.mode == "noise"):
                        self.generate_from_noise(epoch, prt=True, nSeqPrt=3)
                    elif(self.mode == "seq"):
                        self.generate_from_seq(epoch, prt=True, nSeqPrt=3, nsequence=3, seed=X[0,:,0].tolist())

                if(epoch % 200 == 0):
                    self.save_model(epoch)

                self.save_model("last")

                skipDm = "trained"
                if(skipD > 0):
                    skipDm = "skipped"

                self.Dloss.append(d_loss)
                self.Gloss.append(g_loss)
                elapsed_time = time.time() - epoch_start
                print(f"[{epoch+1:03d}/{epochs}] [D: {skipDm}] [D loss: {d_loss[0]:.04f}]  [D lossR: {d_loss_real[0]:.04f} | acc.: {100*d_loss_real[1]:.04f}] [D lossF: {d_loss_fake[0]:.04f} | acc.: {100*d_loss_fake[1]:.04f}] [G loss: {g_loss:.04f}] [{int(elapsed_time / 60)}:{elapsed_time % 60:.02f}]")

        self.plot_loss()
        self.save_model("last")
        if(self.mode == "noise"):
            self.generate_from_noise(epoch, prt=True, nSeqPrt=3)
        elif(self.mode == "seq"):
            self.generate_from_seq(epoch, prt=True, nSeqPrt=3, nsequence=3, seed=X[0,:,0].tolist())

    def named_logs(self, model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def save_model(self, epoch):
        global _model_save
        global _args
        global _dir

        self.D.save(f"{_model_save}{_dir}{_args.name}-discriminator_E{epoch}.h5")
        self.G.save(f"{_model_save}{_dir}{_args.name}-generator_E{epoch}.h5")
        self.gan.save(f"{_model_save}{_dir}{_args.name}-gan_E{epoch}.h5")

    def generate_from_noise(self, save_name=None, prt=False, nSeqPrt=10, nsequence=10, save_path=None, progress=False):
        global _gen_save
        global _args
        global _dir

        if(save_path is None):
            save_path = f'{_gen_save}{_dir}{_args.name}_E{save_name}.str'
        else:
            save_path = f'{save_path}{_args.name}_{save_name}.str'

        print(30*"=")
        full_prediction = []

        predicted = list(np.random.randint(0, self.svocab, self.seqlen))

        prog = None
        if(progress):
            prog = utils.Progbar(nsequence)

        for i in range(nsequence):
            x = np.reshape(predicted, (int(len(predicted) / self.seqlen), self.seqlen, 1))
            #x = x / float(self.svocab)                          # should use normalise function for consistency
            x = neg_normalize(x, self.svocab)
            prediction = self.G.predict(x, verbose=0)

            #generated_sequence = (prediction[0,:,0]*self.svocab).tolist()
            generated_sequence = (neg_denormalize(prediction[0,:,0], self.svocab).tolist())
            tokens = list(map(round, generated_sequence))
            tokens = list(map(int, tokens))
            #generated = " ".join(self.toks2txts[tok] for tok in tokens)
            generated = ""
            if(_args.int):
                for t in tokens:
                    tok = t if t <= (self.svocab-1) else (self.svocab-1)
                    generated += f"{tok} "
            else:
                for t in tokens:                                            # safer
                    tok = t if t <= (self.svocab-1) else (self.svocab-1)
                    generated += f"{self.toks2txts[tok]} "

            if(prt):
                if(i < nSeqPrt):
                    print(generated, end=" ")
                else:
                    prt = False
                    print("...")

            #predicted += tokens
            predicted = list(np.random.randint(0, self.svocab, self.seqlen))
            full_prediction.append(generated)
            if(progress):
                prog.update(i+1)

        if(save_name is not None):
            with open(save_path, 'w') as genFile:
                genFile.write(" ".join(full_prediction))

        print()
        print(30*"=")

    def generate_from_seq(self, save_name=None, prt=False, nSeqPrt=10, nsequence=10, seed=None, save_path=None, progress=False):
        global _gen_save
        global _args
        global _dir

        if(save_path is None):
            save_path = f'{_gen_save}{_dir}{_args.name}_E{save_name}.str'
        else:
            save_path = f'{save_path}{_args.name}_{save_name}.str'

        print(30*"=")
        full_prediction = []
        seeded = "NOISE"

        if(seed is None):
            predicted = list(np.random.randint(0, self.svocab, self.seqlen))
        else:
            predicted = [neg_denormalize(num, self.svocab) for num in seed]
            seeded = "SEED"

        seed = list(map(round, predicted))
        seed = list(map(int, seed))
        seed = self.translate_sequence(seed)

        prog = None
        if(progress):
            prog = utils.Progbar(nsequence)
        else:
            print("/"*30)
            print(f"[{seeded}]:\n{seed}")
            print("\\"*30)

        for i in range(nsequence):
            x = np.reshape(predicted, (int(len(predicted) / self.seqlen), self.seqlen, 1))
            x = neg_normalize(x, self.svocab)
            prediction = self.G.predict(x, verbose=0)
            if(not progress):
                print("\n" + "-"*20 + f"\nRound: {i+1}")
            generated_sequence = (neg_denormalize(prediction[0,:,0], self.svocab).tolist())
            tokens = list(map(round, generated_sequence))
            tokens = list(map(int, tokens))
            generated = ""

            # take only the generated tokens
            newtoks = tokens[-self.stride:]
            if(_args.int):
                for t in newtoks:
                    tok = t if t <= (self.svocab-1) else (self.svocab-1)
                    generated += f"{tok} "
            else:
                for t in newtoks:                                            # safer
                    tok = t if t <= (self.svocab-1) else (self.svocab-1)
                    generated += f"{self.toks2txts[tok]} "

            if(prt):
                if(i < nSeqPrt):
                    print(generated, end=" ")
                else:
                    prt = False
                    print("...")

            if(progress):
                prog.update(i+1)

            # get the full sequence generated to feed back as input of the generator
            predicted = tokens
            full_prediction.append(generated)

        if(save_name is not None):
            with open(save_path, 'w') as genFile:
                genFile.write(" ".join(full_prediction))

        print()
        print(30*"=")

    def generate_samples(self, saved_model, nsamples, nseq, savepath):
        global _args

        self.load_model(saved_model)

        for i in range(nsamples):
            if(self.mode == "noise"):
                print(f"Generating {savepath}{_args.name}_{i}.Str")
                self.generate_from_noise(i, prt=False, nsequence=nseq, save_path=savepath, progress=True)
            elif(self.mode == "seq"):
                print(f"Generating {savepath}{_args.name}_{i}.Str")
                self.generate_from_seq(i, prt=False, nsequence=nseq, save_path=savepath, progress=True)

    def plot_loss(self):
        plt.plot(self.Dloss, c='red')
        plt.plot(self.Gloss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['D', 'G'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('Gan_Loss_per_epoch.png', transparent=True)
        plt.close()

    def plot_models(self):
        global _model_save
        global _dir
        global _args

        utils.plot_model(self.D, to_file=_model_save+f"{_dir}plot_d.png", show_shapes=True, show_layer_names=True)
        utils.plot_model(self.G, to_file=_model_save+f"{_dir}plot_g.png", show_shapes=True, show_layer_names=True)
        utils.plot_model(self.gan, to_file=_model_save+f"{_dir}plot_gan.png", show_shapes=True, show_layer_names=True)

    def translate_sequence(self, sequence):
        t = " ".join(self.toks2txts[tok] for tok in sequence)
        return t

    def load_model(self, checkpoint):
        self.G.load_weights(checkpoint)

    def load_checkpoint(self, checkpoint):
        global _args
        global _model_save

        d = f"{_model_save}{_dir}{_args.name}-discriminator_E{checkpoint}.h5"
        g = f"{_model_save}{_dir}{_args.name}-generator_E{checkpoint}.h5"
        c = f"{_model_save}{_dir}{_args.name}-gan_E{checkpoint}.h5"

        self.G.load_weights(g)
        self.D.load_weights(d)
        self.gan.load_weights(c)


class straceClass():
    def __init__(self, seqlen, batch_size, vocab_size, train_data, test_data, model_only=False):
        self.seqlen = seqlen
        self.batch_size = batch_size
        self.svocab = vocab_size
        self.Closs = []
        self.train_data = train_data

        self.classifier = self.Classifier(self.seqlen)
        self.classifier.summary()

        if(not model_only):
            print("Final building of test dataset...")

            test_dataX, test_datay = [], []

            for data in test_data:
                test_dataX.append(data['seq'])
                test_datay.append(data['label'])

            testdata = tf.data.Dataset.from_tensor_slices((test_dataX, test_datay))
            self.test_data = testdata.shuffle(buffer_size=1000)
            self.test_data = testdata.batch(batch_size)

    def Classifier(self, seqlen):
        c_in = Input(shape=(self.seqlen), name="Classifier_in")

        cl = Embedding(self.svocab, 64, input_length=seqlen)(c_in)

        cl = LSTM(400, kernel_regularizer=regularizers.l2(0.0001), return_sequences=True, activation='tanh')(cl)
        cl = LSTM(400, kernel_regularizer=regularizers.l2(0.0001), activation='tanh')(cl)

        cl = Dense(128, kernel_regularizer=regularizers.l2(0.0001))(cl)
        cl = Dropout(0.4)(cl)
        cl = LeakyReLU(alpha=0.2)(cl)
        cl = BatchNormalization(momentum=0.8)(cl)

        cl = Dense(64, kernel_regularizer=regularizers.l2(0.0001))(cl)
        cl = Dropout(0.4)(cl)
        cl = LeakyReLU(alpha=0.2)(cl)
        cl = BatchNormalization(momentum=0.8)(cl)

        c_out = Dense(1, activation="sigmoid")(cl)

        #lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        #  0.001,
        #  decay_steps=self.batch_size*100,
        #  decay_rate=1,
        #  staircase=False)

        model = Model(c_in, c_out, name="Binary_Classifier")
        #opt = Adam(lr_schedule)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def get_batch(self, dataset, seqlen):

        # get the actual random sequence to build the batch
        randidx = np.random.randint(0, len(dataset), self.batch_size)
        X, y = [], []
        for idx in randidx:
            X.append(dataset[idx]['seq'])
            y.append(dataset[idx]['label'])

        # Do not normalize, we use word embedding
        X = np.reshape(X, (self.batch_size, seqlen))
        y = np.reshape(y, (self.batch_size, 1))

        return X,y

    def train(self, epochs, interval=1):
        global _args
        global _dir

        print("\n" + 65*"=" + "\n" + 27*" " + "Work begins" + 28*" " + "\n" + 65*"=" + "\n")

        self.plot_model()

        if(_args.tb_dir != "NONE"):
            logdir = format_path(_args.tb_dir)
            tb_class = tf.keras.callbacks.TensorBoard(log_dir=f"{logdir}{_args.name}{_dir}classifier", histogram_freq=0, write_graph=True)
            tb_class.set_model(self.classifier)

            tb_classE = tf.keras.callbacks.TensorBoard(log_dir=f"{logdir}{_args.name}{_dir}classifier_eval", histogram_freq=0, write_graph=True)
            tb_classE.set_model(self.classifier)

        for epoch in range(epochs):
            epoch_start = time.time()

            X,y = self.get_batch(self.train_data, self.seqlen)
            c_loss = self.classifier.train_on_batch(X, y)

            if(_args.tb_dir != "NONE"):
                tb_class.on_epoch_end(epoch, self.named_logs(self.classifier, c_loss))

            if(epoch % interval == 0):
                if(epoch % 2 == 0):
                    self.save_model(epoch)

                self.Closs.append(c_loss)
                elapsed_time = time.time() - epoch_start
                print(f"[{epoch+1:03d}/{epochs}] [loss: {c_loss[0]:.04f} | acc.: {100*c_loss[1]:.04f}] [{int(elapsed_time / 60)}:{elapsed_time % 60:.02f}]")

            # model save -- to be used as a checkpoiint
            self.save_model("last")

            test_X, test_y = next(iter(self.test_data))
            c_eval = self.classifier.test_on_batch(test_X, test_y)

            if(_args.tb_dir != "NONE"):
                tb_classE.on_epoch_end(epoch, self.named_logs(self.classifier, c_eval))

            print(f"    [EVAL] [loss: {c_eval[0]:.04f} | acc.: {100*c_eval[1]:.04f}]")

        self.plot_loss()
        self.save_model("last")

        self.eval()

    def eval(self):
        print("\n" + 65*"=" + "\n" + 27*" " + "Eval phase" + 28*" " + "\n" + 65*"=" + "\n")
        c_eval = self.classifier.evaluate(self.test_data, verbose=1)

    def evaluate(self, checkpoint):
        global _args
        global _dir
        global _model_save

        self.classifier.load_weights(f"{_model_save}{_dir}{_args.name}-classifier_E{checkpoint}.h5")
        self.eval()

    def named_logs(self, model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def plot_loss(self):
        plt.plot(self.Closs, c='red')
        plt.title("Classifier Loss per Epoch")
        plt.legend('Classifier')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('Classifier_Loss_per_epoch.png', transparent=True)
        plt.close()

    def plot_model(self):
        global _model_save
        global _dir
        global _args

        utils.plot_model(self.classifier, to_file=_model_save+f"{_dir}plot_d.png", show_shapes=True, show_layer_names=True)

    def save_model(self, epoch):
        global _model_save
        global _args
        global _dir

        self.classifier.save(f"{_model_save}{_dir}{_args.name}-classifier_E{epoch}.h5")

    def load_model(self, checkpoint):
        path = f"{_model_save}{_dir}{_args.name}-classifier_E{checkpoint}.h5"
        self.classifier.load_weights(path)

    def visualise_word_embeddings(self, model, txt2tok):
        self.load_model(model)
        embedding = self.classifier.layers[1]
        print(self.classifier.layers)
        w = embedding.get_weights()[0]

        out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
        out_m = io.open('meta.tsv', 'w', encoding='utf-8')

        for idx, key in enumerate(txt2tok):
            vec = w[idx]
            out_m.write(key +'\n')
            out_v.write('\t'.join([str(x) for x in vec])+'\n')

        out_v.close()
        out_m.close()


def load_gan_dataset():
    global _args
    global _ds_name
    global _ds
    global _ds_meta

    if(_args.gen):
        return None,None,None,None

    _ds_name = _args.dataset
    _ds = format_path(_ds_root + _ds_name)
    _ds_meta = format_path(_ds + "meta_" + _ds_name)

    # get all files from the dataset and store them
    dataset = []
    for file in glob.glob(_ds + "*.str"):
        dataset.append(file)

    metadata = {}
    with open(_ds_meta+_ds_name+'.meta.pkl', 'rb') as meta:
        metadata = pickle.load(meta)

    token_to_text = metadata["tok2txt"]
    text_to_token = metadata["txt2tok"]

    vocab_size = len(token_to_text)

    return dataset, token_to_text, text_to_token, vocab_size

def class_load_full_dataset(path, batch_size, seqlen):
    global _dir
    mfiles_train = get_straces_in_dir(f"{path}train{_dir}malware{_dir}")
    cfiles_train = get_straces_in_dir(f"{path}train{_dir}cleanware{_dir}")

    mfiles_test = get_straces_in_dir(f"{path}test{_dir}malware{_dir}")
    cfiles_test = get_straces_in_dir(f"{path}test{_dir}cleanware{_dir}")

    print("[Building train dataset...]")
    train_dataset = class_build_dataset(mfiles_train, cfiles_train, seqlen)
    print("[Building test dataset...]")
    test_dataset = class_build_dataset(mfiles_test, cfiles_test, seqlen, tiled=True)

    np.random.shuffle(train_dataset)

    return train_dataset, test_dataset

def class_build_dataset(malfiles, cleanfiles, seqlen, tiled=False):
    dataset = []

    ite = 0
    prog = utils.Progbar(len(cleanfiles) + len(malfiles))

    # quick hack: in our current tests generated malware files are 500 tokens long...
    if(not tiled):
        for sample in malfiles:
            with open(sample, 'r') as s:
                lst = list(map(int, s.read().split()))
                dataset.append({'seq':lst, 'label':1})
            ite += 1
            prog.update(ite)
    else:
        for sample in malfiles:
            sequences = get_tiled_list(get_token_list(sample), seqlen)
            for seq in sequences:
                dataset.append({'seq':seq, 'label':1})
            ite += 1
            prog.update(ite)

    # while cleanware token files are 2500 tokens long...
    for sample in cleanfiles:
        sequences = get_tiled_list(get_token_list(sample), seqlen)
        for seq in sequences:
            dataset.append({'seq':seq, 'label':0})
        ite += 1
        prog.update(ite)


    return dataset

def get_tiled_list(lst, length):
    tlst = []
    for i in range(0, len(lst) - length):
        tlst.append(lst[i:i+length])
    return tlst

def get_token_list(file):
    tokens = []
    with open(file, 'r') as f:
        tokens = list(map(int, f.read().split()))
    return tokens

def get_straces_in_dir(dir):
    tmp = []

    for file in glob.glob(dir + "*.str"):
        tmp.append(file)
    return tmp

def normalize(val, size):
    return val / float(size)

def neg_normalize(val, svocab):
    return (val - float(svocab) / 2) / (float(svocab) / 2)

def neg_denormalize(val, svocab):
    return (val * float(svocab) / 2) + (float(svocab) / 2)

def format_path(path):
    if(os.name == 'nt'):
        if(path[-1:] == "\\"):
            return path
        else:
            return path + "\\"
    else:
        if(path[-1:] == "/"):
            return path
        else:
            return path + "/"

def setup_env(name):
    global _dir

    if(os.name == 'nt'):
        _dir = "\\"

    modeldir = f'.{_dir}model{_dir}' + name
    gendir = f'.{_dir}generated{_dir}' + name

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if not os.path.exists(gendir):
        os.makedirs(gendir)

    return modeldir, gendir

def get_gan(seqlen, dataset, txt2tok, tok2txt, vocab_size, stride=10):
    global _args

    gan = None
    if(_args.noise):
        gan = straceGan(seqlen=seqlen, dataset=dataset, txt2tok=txt2tok, tok2txt=tok2txt, svocab=vocab_size, mode="noise")
    elif(_args.seq):
        gan = straceGan(seqlen=seqlen, dataset=dataset, txt2tok=txt2tok, tok2txt=tok2txt, svocab=vocab_size, stride=_args.stride, mode="seq")

    return gan

def args_inquisitor():
    global _args
    parser = argparse.ArgumentParser(description="Train a gan to generate Android malware strace file.\nOr a classifier to detect Android malware.\nOr load a model and generate samples")
    parser.add_argument('--dataset', help='Name of the dataset to use in seq/noise mode. Sometimes needed for other modes, conditions not fully implemented')
    parser.add_argument('-l', '--seqlen', type=int, help="Length of sequence to be generated. Needed even in generation mode, to build the correct model")
    parser.add_argument('-s', '--stride', type=int, default=10, help="Number of tokens to generate and add to the input sequence to create a new sequence -- ONLY FOR SEQ MODE")
    parser.add_argument('-n', '--name', help='Save name of the run, will be used in the generated samples and h5 models save files')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('-t', '--tb_dir', help='Tensorboad log dir. Activate tensorboard logging only if this option is set', default="NONE")
    parser.add_argument('-v', '--vocab', help='Override vocabulary size, or specify size for training the classifier. Not fully implemented, use a vocab file instead.', type=int, default=-1)
    parser.add_argument('--vocab_file', help='Load vocabulary data from a pickle file, overrides vocabulary size value, and tokens dictionaries', type=str, default=None)
    parser.add_argument('-b', '--batch', help='Batch size -- used as number of samples to generated in gen mode', type=int, default=32)
    parser.add_argument('--weights', help='Name of the saved model weights to load in gen mode.')
    parser.add_argument('--gen_output', help='Folder where the generated samples in gen mode data will be saved')
    parser.add_argument("--gen", action="store_true", default=False, help="Load a generator model and generate data")
    parser.add_argument("--n_seq", type=int, help="Number of sequence to generate. n_seq * seqlen == length of generated sample")
    parser.add_argument('--checkpoint', help='Resume training at this epoch. h5 file loaded from save_name parameters and default folders structure', type=int, default=None)
    parser.add_argument('-i', '--int', help='Generate tokens instead of words', action='store_true', default=False)
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluate the model at a given checkpoint")
    parser.add_argument("--vis", action="store_true", default=False, help="Generate visualisation files for word embeddings from the specified checkpoint")


    mutex = parser.add_mutually_exclusive_group()
    mutex.add_argument("--noise", action="store_true", default=False, help="use noise to generate sequences")
    mutex.add_argument("--seq", action="store_true", default=False, help="use true sequence to generate the next part of the sequence")
    mutex.add_argument("--classTrain", action="store_true", default=False, help="Train the strace classifier")
    mutex.add_argument("--classInfer", action="store_true", default=False, help="Ask the oracle")

    _args = parser.parse_args()

if(__name__ == '__main__'):
    tf.keras.backend.set_floatx('float64')      # to solve a weird casting issue
    args_inquisitor()

    _ds_root = f".{_dir}dataset{_dir}"

    # Not model specific vars
    seqlen = _args.seqlen
    batchSize = _args.batch
    epochs = _args.epochs

    _model_save, _gen_save = setup_env(_args.name)

    tok2txt, txt2tok = {}, {}
    vocab_size = _args.vocab

    if(_args.vocab_file is not None):
        vocab_data = {}
        with open(_args.vocab_file, 'rb') as vfile:
            vocab_data = pickle.load(vfile)
        tok2txt = vocab_data['tok2txt']
        txt2tok = vocab_data['txt2tok']
        vocab_size = vocab_data['size']

    if(_args.noise or _args.seq):
        if(_args.vocab_file is None):
            dataset, tok2txt, txt2tok, vocab_size = load_gan_dataset()
        else:
            dataset, _, _, _ = load_gan_dataset()

        gan = get_gan(seqlen=seqlen, dataset=dataset, txt2tok=txt2tok, tok2txt=tok2txt, vocab_size=vocab_size, stride=_args.stride)

        if(_args.gen):
            gan.generate_samples(saved_model=_args.weights, nsamples=_args.batch, nseq=_args.n_seq, savepath=format_path(_args.gen_output))
        else:
            if(_args.checkpoint is not None):
                gan.load_checkpoint(_args.checkpoint)
            gan.train(epochs=epochs, batch_size=batchSize, full_seq_training=True, e=_args.checkpoint)

    elif(_args.classTrain):
        train_data, test_data = class_load_full_dataset(f".{_dir}dataset{_dir}classify{_dir}", batchSize, seqlen)
        classifier = straceClass(seqlen=seqlen, batch_size=batchSize, vocab_size=vocab_size, train_data=train_data, test_data=test_data, model_only=_args.vis)
        if(not _args.eval and not _args.vis):
            classifier.train(epochs=epochs)
        elif(_args.eval):
            classifier.evaluate(checkpoint=_args.checkpoint)
        elif(_args.vis):
            classifier.visualise_word_embeddings(_args.checkpoint, txt2tok)
    elif(_args.classInfer):
        print("MODE NOT IMPLEMENTED YET")
    else:
        print("You must select a mode: --seq or --noise or --classTrain")
