from flask import Flask, render_template, request, redirect, url_for
import os
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",data="ここに読み上げられるテキストがきます")

@app.route("/upload_photo", methods=['GET','POST'])
def upload_photo():
    if request.method == 'POST':
        if "photo" in request.json:
            photo_data = request.json['photo']
            save_photo(photo_data, 'static/photos', 'snapshot.png')
            return "写真が送信されました"
        else:
            return "写真が見つかりません"
    else:
        COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        #以下略
        #画像中の物体検出
        #必要なライブラリのimport
        #Colabには標準でインストールされているのでinstallの必要はなし

        import torch #深層学習用ライブラリの一つであるpytorchをimport
        import PIL #画像処理用ライブラリの一つPILをimport
        from PIL import Image #PILからImageをimport
        import torchvision #pytorchの画像処理用ライブラリをimport
        from torchvision import transforms #画像処理用ライブラリからtransformsをimport
        import cv2 #画像処理用ライブラリ
        import matplotlib.pyplot as plt #グラフや画像を表示するためのライブラリをimport

        # 画像を読み込む
        frame_raw = cv2.imread('./static/photos/snapshot.png')

        # frame_rawがNoneでないことを確認
        if frame_raw is not None:
            # BGRの順番からRGBの順番に変換
            frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)

            # NumPy配列からPIL形式に変換
            image = Image.fromarray(frame)

            # 変換後の画像を使って処理を続ける

        else:
            print("画像の読み込みに失敗しました。")

        #モデルをダウンロード

        #Faster R-CNNはこちら
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        #Mask R-CNNを試したい場合は、一行下のコードのコメントアウト(#)を外す
        #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        X = []  #x座標格納用
        Y = []  #y座標格納用
        with torch.no_grad():
            # 推論に使うデバイスを選択（GPUを使用する場合は torch.device('cuda')）
            device = torch.device('cpu')
            transform = transforms.Compose([transforms.ToTensor()])  # PILをTensorに変換するためのインスタンスを用意
            inputs = transform(image)  # PIL → Tensor
            inputs = inputs.unsqueeze(0).to(device)  # デバイスに入力
            model.eval()  # モデルを推論モードに切り替え
            outputs = model(inputs)  # モデルに推論させて結果を受け取る
            for i in range(len(outputs[0]['boxes'])):
                x0 = int(outputs[0]['boxes'][i][0])  # BBの左上の点のx座標
                y0 = int(outputs[0]['boxes'][i][1])  # BBの左上の点のy座標
                x1 = int(outputs[0]['boxes'][i][2])  # BBの右下の点のx座標
                y1 = int(outputs[0]['boxes'][i][3])  # BBの右下の点のy座標
                # confidence(モデルがその推論にどのくらい自信があるか)が0.7以上だったら
                if outputs[0]['scores'][i] >= 0.3:
                    class_num = outputs[0]['labels'][i]
                    class_name = COCO_INSTANCE_CATEGORY_NAMES[class_num]
                    if class_name=="bottle":
                        X.append(x0)
                        Y.append(y0)
                        bbox = cv2.rectangle(frame_raw, (x0, y0), (x1, y1), (0, 0, 255), 3, cv2.LINE_4)  # BBを表示
                        conf = float(outputs[0]['scores'][i])
                        conf = round(conf,4)
                        label = class_name + str(conf)
                        print(">>"+label)
                        print(((x0+x1)/2),((y0+y1)/2))
                        bbox = cv2.putText(bbox,label,(x0,y0),cv2.FONT_HERSHEY_COMPLEX,2.1,(0,128,0),2 ) #ラベルを表示
                        # 画像読み込みのパス
                        img = cv2.imread("./static/photos/snapshot.png")
                        # サンプル１の画像
                        # image_1 = img[y0:y1, x0:x1]
                        # cv2.imwrite("./static//test_photo/"+str(i)+".jpg", image_1)

            after_X = sorted(X)
            after_Y = sorted(Y)
            # 結果を表示
            # plt.figure(figsize=(12, 9))  # 表示する画像のサイズを決定
            # plt.imshow(cv2.cvtColor(bbox, cv2.COLOR_BGR2RGB))  # cv2はBGRだがpltはRGB
            # plt.axis("off")  # グラフの目盛りが入るのを防ぐ
            # plt.show()  # 結果の画像を表示

        from keras.utils import load_img, img_to_array
        from keras.models import model_from_json
        import numpy as np
        from pathlib import Path
        import os
        # 画像が保存されているルートディレクトリのパス
        root_dir = Path('__file__').resolve().parent

        # 商品名
        categories = ["acc", "tea", "kola", "mugi", "nattyan", "grape"]
        # 保存したモデルの読み込み
        model = model_from_json(
            open(root_dir /  "tea_predict4.json").read())
        # 保存した重みの読み込み
        model.load_weights(root_dir / "tea_predict4.hdf5")

        user_select = 1

        # 画像を読み込む
        #ここをfor文で回してcut_photo内の写真を判定するような形でやる
        photo_dir = (root_dir / 'static//test_photo') 
        files = os.listdir(photo_dir)
        for idx, file in enumerate(files):
            img_path = (photo_dir / file)
            img = load_img(img_path, target_size=(250, 250, 3))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # 予測
            features = model.predict(x)

            # 予測結果によって処理を分ける
            if features[0, 0] == 1:
                print("acc")

            elif features[0, 1] == 1:
                print("tea")

            elif features[0, 2] == 1:
                data = 'コーラを発見'
                x_idx = X[idx]  
                y_idx = Y[idx]
                
                for i in range(len(X)):
                    if x_idx == after_X[i]:
                        x = i
                for j in range(len(Y)):
                    if y_idx == after_Y[j]:
                        y = j
                    
                message = ('コーラは{}段目{}列目にあります'.format(int(y/3 + 1), int(x/3 + 1)))
                

            else:
                for i in range(0, 6):
                    if features[0, i] == 1:
                        cat = categories[i]
                        message = cat
                        print(message)

            return render_template("index.html", data=message)

def save_photo(photo_data, save_dir, file_name):
    # base64エンコードされた写真データをデコード
    photo_bytes = photo_data.split(',')[1].encode('utf-8')

    # ファイルの保存先ディレクトリを作成
    os.makedirs(save_dir, exist_ok=True)

    # ファイルパスを結合
    file_path = os.path.join(save_dir, file_name)

    # 写真をファイルとして保存
    with open(file_path, 'wb') as f:
        f.write(base64.b64decode(photo_bytes))

if __name__ == "__main__":
    app.run(debug=True)