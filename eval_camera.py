import cv2
import numpy as np


def capture_video(camera_index=0, is_gray=False):
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

        if is_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # フレーム表示
        cv2.putText(
            frame,
            text='[Shape]: {} / [FPS]: {:.2f}'.format(frame.shape, fps),
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


if __name__ == '__main__':
    capture_video(is_gray=True)
    