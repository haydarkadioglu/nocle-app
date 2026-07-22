# %%
from google.colab import drive
drive.mount('/content/drive')
import os

os.chdir("/content/drive/MyDrive/python/noisy")

# %%
import tensorflow as tf
from keras.layers import Conv1D,Conv1DTranspose,Concatenate,Input,Cropping1D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
import numpy as np
import IPython.display
import glob
from tqdm import tqdm
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%
clean_train = np.load('data/12k-batch/clean_train.npy')
pure = np.load('data/12k-batch/pure_part3.npy')

print(f"Clean Train Shape: {clean_train.shape}")
print(f"Noisy Train Shape: {pure.shape}")


# %%
# Stereo'yu Mono'ya çevir (eğer stereo ise)
if clean_train.ndim == 3 and clean_train.shape[-1] == 2:
    clean_train = np.mean(clean_train, axis=-1)
if pure.ndim == 3 and pure.shape[-1] == 2:
    pure = np.mean(pure, axis=-1)


# %%

pure = pure[..., np.newaxis]


# Boyutları eşitle
min_len = min(clean_train.shape[0], pure.shape[0])
min_sample_len = min(clean_train.shape[1], pure.shape[1])

clean_train = clean_train[:min_len, :min_sample_len]
pure = pure[:min_len, :min_sample_len]

# Gürültüyü %30 oranında karıştır
noisy_mixed = clean_train + 0.5 * pure



# %%
print(f"Clean Train Shape: {clean_train.shape}")
print(f"Noisy Train Shape: {pure.shape}")

# %%
import numpy as np
from IPython.display import Audio

# Örnekleme oranı
sampling_rate = 16000

# Boyutları kontrol edip sıkıştır
clean_data = clean_train.squeeze()
noisy_data = noisy_mixed.squeeze()

# İlk 10 örneği birleştir
clean_concat = np.concatenate(clean_data[4050:4100], axis=0)
noisy_concat = np.concatenate(noisy_data[4050:4100], axis=0)

# Ses verisini -1.0 ile 1.0 arasına normalize et
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val != 0 else audio

clean_concat = normalize_audio(clean_concat)
noisy_concat = normalize_audio(noisy_concat)

# 🎧 Oynat
print("🎧 Clean Audio (first 10 samples concatenated):")
display(Audio(clean_concat, rate=sampling_rate))

print("🎧 Noisy Audio (first 10 samples concatenated):")
display(Audio(noisy_concat, rate=sampling_rate))


# %%
clean_combined = np.concatenate((np.load('data/12k-batch/clean_train.npy'),
                                #  np.load('data/12k-batch/clean_train2.npy'),
                                 ), axis=0)

noisy_combined = np.concatenate((np.load('data/12k-batch/noisy_train.npy'),
                                #  np.load('data/12k-batch/noisy_train2.npy')
                                 ), axis=0)

print(f"Clean Combined Shape: {clean_combined.shape}")
print(f"Noisy Combined Shape: {noisy_combined.shape}")

# %%


# %%


# %%


# %%
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GRU, Bidirectional
from tensorflow.keras.layers import Conv1DTranspose, Concatenate, Dropout, TimeDistributed
from tensorflow.keras.models import Model

def build_generator(input_shape=(12000, 1)):
    inp = Input(shape=input_shape)

    # --- Encoder ---
    x = Conv1D(16, 15, strides=2, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c1 = x

    x = Conv1D(32, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c2 = x

    x = Conv1D(64, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c3 = x

    x = Conv1D(128, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c4 = x

    x = Conv1D(256, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    c5 = x

    # --- Bottleneck with Bidirectional GRU ---
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)


    x = TimeDistributed(Dense(128, activation='relu'))(x)


    # --- Decoder ---
    x = Conv1DTranspose(256, 15, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([c5, x])

    x = Conv1DTranspose(128, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([c4, x])

    x = Conv1DTranspose(64, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([c3, x])

    x = Conv1DTranspose(32, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([c2, x])

    x = Conv1DTranspose(16, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([c1, x])

    x = Conv1DTranspose(8, 15, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # inp ile tekrar concatenate ETMİYORUZ artık!

    out = Conv1DTranspose(1, 15, strides=1, padding='same', activation='linear')(x)

    model = Model(inputs=inp, outputs=out, name="GRU_Generator")
    return model


# %%
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU, Flatten, Dense
from tensorflow.keras.models import Model

def build_discriminator(input_shape=(12000, 1)):
    inp = Input(shape=input_shape)

    x = Conv1D(16, 15, strides=4, padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(32, 15, strides=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(64, 15, strides=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv1D(128, 15, strides=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='sigmoid')(x)  # 0 (fake) or 1 (real)

    model = Model(inputs=inp, outputs=out, name="Discriminator")
    return model


# %%
from keras.models import load_model
generator = load_model('dg_gru_original-100.hdf5', compile=False)



# %%



# generator = build_generator(input_shape=(12000, 1))
discriminator = build_discriminator(input_shape=(12000, 1))

generator.summary()
discriminator.summary()


# %%
BATCH_SIZE = 32
EPOCHS = 25
learning_rate = 0.001
lambda_l1 = 100  # L1 loss katkısı
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output, clean_audio, generated_audio):
    adv_loss = bce(tf.ones_like(fake_output), fake_output)
    l1_loss = tf.reduce_mean(tf.abs(clean_audio - generated_audio))
    return adv_loss + lambda_l1 * l1_loss
gen_optimizer = Adam(learning_rate, beta_1=0.5)
disc_optimizer = Adam(learning_rate, beta_1=0.5)

@tf.function
def train_step(noisy_audio, clean_audio):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_audio = generator(noisy_audio, training=True)

        real_output = discriminator(clean_audio, training=True)
        fake_output = discriminator(generated_audio, training=True)

        gen_loss = generator_loss(fake_output, clean_audio, generated_audio)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, generated_audio  # <-- Burada döndür




def train_numpy(noisy_data, clean_data, epochs=100, batch_size=32):
    # Kayıt için boş listeler
    history = {
        "epoch": [],
        "gen_loss": [],
        "disc_loss": [],
        "mae": [],
        "mape": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        gen_losses = []
        disc_losses = []
        mae_list = []
        mape_list = []

        num_samples = noisy_data.shape[0]
        indices = np.random.permutation(num_samples)  # Shuffle manually

        for i in tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}", ncols=100):
            batch_indices = indices[i:i+batch_size]

            noisy_batch = tf.convert_to_tensor(noisy_data[batch_indices], dtype=tf.float32)
            clean_batch = tf.convert_to_tensor(clean_data[batch_indices], dtype=tf.float32)

            g_loss, d_loss, generated_audio = train_step(noisy_batch, clean_batch)

            gen_losses.append(g_loss.numpy())
            disc_losses.append(d_loss.numpy())

            # MAE ve MAPE hesapla
            mae = tf.reduce_mean(tf.abs(clean_batch - generated_audio)).numpy()
            mape = tf.reduce_mean(tf.abs((clean_batch - generated_audio) / (clean_batch + 1e-7))).numpy() * 100
            mae_list.append(mae)
            mape_list.append(mape)

        # Epoch sonuçlarını kaydet
        history["epoch"].append(epoch + 1)
        history["gen_loss"].append(np.mean(gen_losses))
        history["disc_loss"].append(np.mean(disc_losses))
        history["mae"].append(np.mean(mae_list))
        history["mape"].append(np.mean(mape_list))

        print(f"[Epoch {epoch + 1}] GenLoss: {np.mean(gen_losses):.4f} | DiscLoss: {np.mean(disc_losses):.4f} | MAE: {np.mean(mae_list):.6f} | MAPE: {np.mean(mape_list):.2f}%")

    return history


def train(dataset, val_dataset, epochs):
    history = {
        "epoch": [],
        "gen_loss": [],
        "disc_loss": [],
        "mae": [],
        "mape": [],
        "val_mae": [],
        "val_mape": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        gen_losses = []
        disc_losses = []
        mae_list = []
        mape_list = []

        step_bar = tqdm(enumerate(dataset), total=len(dataset), desc=f"Epoch {epoch + 1} [Train]", ncols=100)

        for step, (noisy_batch, clean_batch) in step_bar:
            g_loss, d_loss, generated_audio = train_step(noisy_batch, clean_batch)

            gen_losses.append(g_loss.numpy())
            disc_losses.append(d_loss.numpy())

            # MAE ve MAPE hesapla (train)
            mae = tf.reduce_mean(tf.abs(clean_batch - generated_audio)).numpy()
            mape = tf.reduce_mean(tf.abs((clean_batch - generated_audio) / (clean_batch + 1e-7))).numpy() * 100
            mae_list.append(mae)
            mape_list.append(mape)

        # Validation MAE ve MAPE hesapla (tqdm bar ile)
        val_mae_list = []
        val_mape_list = []

        val_bar = tqdm(val_dataset, desc=f"Epoch {epoch + 1} [Val]  ", ncols=100)
        for val_noisy, val_clean in val_bar:
            val_generated = generator(val_noisy, training=False)
            val_mae = tf.reduce_mean(tf.abs(val_clean - val_generated)).numpy()
            val_mape = tf.reduce_mean(tf.abs((val_clean - val_generated) / (val_clean + 1e-7))).numpy() * 100
            val_mae_list.append(val_mae)
            val_mape_list.append(val_mape)

        # Sonuçları kaydet
        history["epoch"].append(epoch + 1)
        history["gen_loss"].append(np.mean(gen_losses))
        history["disc_loss"].append(np.mean(disc_losses))
        history["mae"].append(np.mean(mae_list))
        history["mape"].append(np.mean(mape_list))
        history["val_mae"].append(np.mean(val_mae_list))
        history["val_mape"].append(np.mean(val_mape_list))

        # Epoch sonucu yazdır
        print(f"[Epoch {epoch + 1}] "
              f"GenLoss: {np.mean(gen_losses):.4f} | DiscLoss: {np.mean(disc_losses):.4f} | "
              f"Train MAE: {np.mean(mae_list):.6f} | Train MAPE: {np.mean(mape_list):.2f}% | "
              f"Val MAE: {np.mean(val_mae_list):.6f} | Val MAPE: {np.mean(val_mape_list):.2f}%")

    return history




# %%
# history = train_numpy(pure[:2000], clean_train[:2000], epochs=EPOCHS, batch_size=BATCH_SIZE)


# %%
# dataset = tf.data.Dataset.from_tensor_slices((noisy_mixed, clean_train))
# dataset = dataset.shuffle(100).batch(BATCH_SIZE)

# %%
from sklearn.model_selection import train_test_split

# İlk olarak veriyi %70 train, %30 (val + test) olarak ayır
noisy_train, noisy_temp, clean_train_split, clean_temp = train_test_split(
    noisy_mixed, clean_train, test_size=0.3, random_state=42
)

# Şimdi val ve test'i %50-%50 ayır (0.5 * 0.3 = %15 val, %15 test olacak)
noisy_val, noisy_test, clean_val, clean_test = train_test_split(
    noisy_temp, clean_temp, test_size=0.8, random_state=42
)
train_dataset = tf.data.Dataset.from_tensor_slices((noisy_train, clean_train_split)).shuffle(500).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((noisy_val, clean_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((noisy_test, clean_test)).batch(BATCH_SIZE)


# %%
history = train(train_dataset, val_dataset, epochs=EPOCHS)


# %%
history2 = train(train_dataset, val_dataset, epochs=EPOCHS)


# %%
history3 = train(train_dataset, val_dataset, epochs=EPOCHS)


# %%
history4 = train(train_dataset, val_dataset, epochs=EPOCHS)


# %%
history5 = train(train_dataset, val_dataset, epochs=EPOCHS)


# %%
history6 = train(train_dataset, val_dataset, epochs=EPOCHS)


# %%
# Generator modelini kaydet
generator.save("dg_gru_original-175.hdf5")


# %%
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = history["epoch"]

    # Generator Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["gen_loss"], label="Generator Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Discriminator Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["disc_loss"], label="Discriminator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # MAE - Training vs Validation
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["mae"], label="Train MAE", color="green")
    plt.plot(epochs, history["val_mae"], label="Val MAE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Mean Absolute Error (MAE)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # MAPE - Training vs Validation
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["mape"], label="Train MAPE", color="green")
    plt.plot(epochs, history["val_mape"], label="Val MAPE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE (%)")
    plt.title("Mean Absolute Percentage Error (MAPE)")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_training_history(history)


# %%
import csv

def save_history_to_csv(history, filename="history3.csv"):
    # Başlıkları tanımla
    headers = ["Epoch", "GenLoss", "DiscLoss", "Train MAE", "Train MAPE", "Val MAE", "Val MAPE"]

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                round(history["gen_loss"][i], 4),
                round(history["disc_loss"][i], 4),
                round(history["mae"][i], 6),
                round(history["mape"][i], 2),
                round(history["val_mae"][i], 6),
                round(history["val_mape"][i], 2)
            ])

    print(f"[✓] Training history saved as CSV to: {filename}")

save_history_to_csv(history=history5, filename="history5.csv")




# %%
def test_model(test_dataset):
    mae_list = []
    mape_list = []

    print("\n🔍 Running test evaluation...")
    for step, (noisy_batch, clean_batch) in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Testing", ncols=100):
        # Üretilen (tahmin edilen) sesi oluştur
        generated_audio = generator(noisy_batch, training=False)

        # MAE ve MAPE hesapla
        mae = tf.reduce_mean(tf.abs(clean_batch - generated_audio)).numpy()
        mape = tf.reduce_mean(tf.abs((clean_batch - generated_audio) / (clean_batch + 1e-7))).numpy() * 100

        mae_list.append(mae)
        mape_list.append(mape)

    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)

    print(f"\n📊 Test MAE: {avg_mae:.6f}")
    print(f"📊 Test MAPE: {avg_mape:.2f}%")

    return avg_mae, avg_mape

# %%
# Train Part 5

test_mae, test_mape = test_model(test_dataset)


# %% [markdown]
# Training Part 1
# 
# 🔍 Running test evaluation...
# Testing: 100%|████████████████████████████████████████████████████| 300/300 [02:45<00:00,  1.81it/s]
# 📊 Test MAE: 0.011288
# 📊 Test MAPE: 9975.16%
# _____________________________________________________________

# %% [markdown]
# Train Part 2
# 
# 🔍 Running test evaluation...
# Testing: 100%|████████████████████████████████████████████████████| 300/300 [02:49<00:00,  1.77it/s]
# 📊 Test MAE: 0.010215
# 📊 Test MAPE: 9623.50%
# _________________________________________

# %% [markdown]
# Train Part 3
# 
# 🔍 Running test evaluation...
# Testing: 100%|████████████████████████████████████████████████████| 300/300 [02:48<00:00,  1.78it/s]
# 📊 Test MAE: 0.012078
# 📊 Test MAPE: 7149.88%
# 
# ________________________________________

# %% [markdown]
#  Train Part 4
# 
# 🔍 Running test evaluation...
# Testing: 100%|████████████████████████████████████████████████████| 300/300 [02:47<00:00,  1.79it/s]
# 📊 Test MAE: 0.008031
# 📊 Test MAPE: 5947.02%
# ______________________________________________

# %% [markdown]
# Training Part 5
# 
# 🔍 Running test evaluation...
# Testing: 100%|████████████████████████████████████████████████████| 300/300 [02:47<00:00,  1.79it/s]
# 📊 Test MAE: 0.008357
# 📊 Test MAPE: 5941.44%
# ______________________________________

# %%


# %%
import IPython.display as ipd
noisy_samples = []
clean_samples = []
generated_samples = []

for noisy_batch, clean_batch in test_dataset.take(1):  # tek batch al
    generated_batch = generator(noisy_batch, training=False)

    for i in range(min(20, noisy_batch.shape[0])):
        noisy = noisy_batch[i].numpy()
        clean = clean_batch[i].numpy()
        generated = generated_batch[i].numpy()

        # Stereo ise mono'ya çevir
        if len(noisy.shape) == 2:
            noisy = np.mean(noisy, axis=1)
            clean = np.mean(clean, axis=1)
            generated = np.mean(generated, axis=1)

        noisy_samples.append(noisy)
        clean_samples.append(clean)
        generated_samples.append(generated)


# %%
# Birleştir (art arda diz)
noisy_concat = np.concatenate(noisy_samples)
clean_concat = np.concatenate(clean_samples)
generated_concat = np.concatenate(generated_samples)
print("🎙️ 20 Gürültülü Giriş (Yan yana dizilmiş):")
ipd.display(ipd.Audio(noisy_concat, rate=16000))

print("🔈 20 Temiz Orijinal Ses:")
ipd.display(ipd.Audio(clean_concat, rate=16000))

print("🎧 20 Modelin Tahmin Ettiği Ses:")
ipd.display(ipd.Audio(generated_concat, rate=16000))



# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
# from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, GRU, Bidirectional, LSTM
# from tensorflow.keras.layers import Conv1DTranspose, Concatenate, Dropout
# from tensorflow.keras.models import Model

# def build_generator(input_shape=(48000, 1)):
#     inp = Input(shape=input_shape)

#     # --- Encoder (güçlü filtreler) ---
#     x = Conv1D(16, 15, strides=2, padding='same')(inp)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     c1 = x

#     x = Conv1D(32, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     c2 = x

#     x = Conv1D(64, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     c3 = x

#     x = Conv1D(128, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     c4 = x

#     x = Conv1D(256, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     c5 = x

#     # --- Bottleneck with stronger RNN ---
#     x = Bidirectional(LSTM(64, return_sequences=True))(x)
#     x = Dropout(0.3)(x)
#     x = Bidirectional(LSTM(128, return_sequences=True))(x)
#     x = Dropout(0.3)(x)
#     x = Bidirectional(LSTM(64, return_sequences=True))(x)
#     x = Dropout(0.3)(x)

#     # --- Decoder (matching filtre sayıları) ---
#     x = Conv1DTranspose(256, 15, strides=1, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Concatenate()([c5, x])

#     x = Conv1DTranspose(128, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Concatenate()([c4, x])

#     x = Conv1DTranspose(64, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Concatenate()([c3, x])

#     x = Conv1DTranspose(32, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Concatenate()([c2, x])

#     x = Conv1DTranspose(16, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Concatenate()([c1, x])

#     x = Conv1DTranspose(8, 15, strides=2, padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Concatenate()([inp, x])

#     out = Conv1DTranspose(1, 15, strides=1, padding='same', activation='linear')(x)


#     model = Model(inputs=inp, outputs=out, name="Generator")
#     return model



