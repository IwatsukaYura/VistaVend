# ラベリングによる学習/検証データの準備

from PIL import Image
import os
import glob
import numpy as np
import random
import math
from keras import layers, models, optimizers
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pickle
from keras.utils import np_utils
from keras.preprocessing import image
from pathlib import Path

# 画像が保存されているルートディレクトリのパス
root_dir = Path(__file__).resolve().parent
# 商品名
categories = ["acc", "tea", "kola", "mugi", "nattyan", "grape"]

# 画像データ用配列
X = []
# ラベルデータ用配列
Y = []

# 画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数


def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X, dtype=object), np.array(Y, dtype=object)

# 渡された画像データを読み込んでXに格納し、また、
# 画像データに対応するcategoriesのインデックスをYに格納する関数


def add_sample(cat, fname):
    img = Image.open(fname)  # 画像ファイルを開いて識別
    img = img.convert("RGB")  # 画像ファイルをRGBに変換
    img = img.resize((250, 250))  # リサイズ
    data = np.asarray(img)  # 画像データを配列に格納
    X.append(data)  # Xに画像データを代入
    Y.append(cat)  # Yにカテゴリーラベルを代入


# 全データ格納用配列
allfiles = []

# カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for idx, cat in enumerate(categories):
    image_dir = (root_dir / cat)  # 例：ルートディレ/お～いお茶
    # お～いお茶ディレクトリに含まれるjpg画像を配列で取得しfilesに代入
    files = glob.glob(os.path.join(image_dir, "*.jpg"))

    # filesに含まれる画像データ一枚ずつを全データ格納用配列にインデックスとともに代入
    for f in files:
        allfiles.append((idx, f))


# シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)  # 画像データを8:2に分割している
train = allfiles[0:th]  # 8割を学習用データ
test = allfiles[th:]  # 2割を検証用データ
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)

# # データを保存する（データの名前を「data.npy」としている）
with open('model.pickle', mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
    pickle.dump(xy, f)


# モデルの構築
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(250, 250, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(6, activation="sigmoid"))  # 分類先の種類分設定

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"],
              run_eagerly=True)


nb_classes = len(categories)

X_train, X_test, y_train, y_test = np.load(
    root_dir / "model.pickle", allow_pickle=True)

# データの正規化
X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

# kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)



# モデルの学習
model = model.fit(X_train,
                  y_train,
                  epochs=7,
                  batch_size=6,
                  validation_data=(X_test, y_test))


# 学習結果を表示
acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('seido')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss')

# モデルの保存

json_string = model.model.to_json()
open(root_dir / "tea_predict4.json", 'w').write(json_string)

# 重みの保存

hdf5_file = (root_dir / "tea_predict4.hdf5")
model.model.save_weights(hdf5_file)