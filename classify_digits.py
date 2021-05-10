import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 学習済みモデルを読込み
model = load_model('./model/trained_mnist_v1.0.h5')


def capture_video(camera_index=0):
    """ カメラ映像をキャプチャ
        ※ キーボード：q 入力で終了
    """
    # キャプチャオブジェクト取得
    capture = cv2.VideoCapture(0)
    tm = cv2.TickMeter()
    tm.start()
    count, max_count, fps = 0, 10, 0
    while True:
        _r, frame = capture.read()
        # FPS の計算処理
        if count == max_count:
            tm.stop()
            fps = max_count / tm.getTimeSec()
            tm.reset()
            tm.start()
            count = 0
        # 認識エリア（赤枠:600x600）を表示
        w_center, h_center = calculate_center(frame)
        cv2.rectangle(
            frame, (w_center - 300, h_center - 300), (w_center + 300, h_center + 300), (0, 0, 255)
        )
        # 数字の識別
        predict = classify_digits(frame)
        digits = np.argmax(predict)
        confidence = predict[0][digits]
        # フレーム表示
        cv2.putText(
            frame,
            text='[FPS]: {:.2f} / [Predict]: {} ({:.5f})'.format(fps, digits, confidence),
            # text='[FPS]: {:.2f} / [Predict]: {}'.format(fps, digits),
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=2
        )
        cv2.imshow('Capture', frame)
        count += 1
        # 表示ダイアログでキー: q が押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()


def classify_digits(frame):
    """ フレーム内の手書き数字を識別

    """
    w_center, h_center = calculate_center(frame)
    # 画像トリミング
    d = 300 - 1
    trim  = frame[(h_center - d):(h_center + d), (w_center - d):(w_center + d)]
    gray  = cv2.cvtColor(trim, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    inv   = cv2.bitwise_not(th)
    blur  = cv2.GaussianBlur(inv, (9, 9), 0)
    rsz   = cv2.resize(blur, (28, 28), cv2.INTER_CUBIC)
    rsh   = rsz.reshape(1, 28, 28)
    _x = np.array(rsh) / 255
    predict = model.predict(_x, batch_size=1)
    return predict


def calculate_center(frame):
    """ フレーム画像の中心位置（センター）を計算

    """
    # 画像のサイズ取得
    h, w, _ = frame.shape[:3]
    w_center = w // 2
    h_center = h // 2
    return w_center, h_center


if __name__ == '__main__':
    capture_video()
    