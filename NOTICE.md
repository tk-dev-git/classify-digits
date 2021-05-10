### VSCode のターミナルから Mac 搭載カメラを使用
    - （結論）VSCode を直接起動してターミナルから cv2 等を実行してもセキュリティ制限で起動しない（abort）
    - 現状の回避策
        1. iTerm 等のカメラ使用許可ダイアログが表示される他ターミナルソフトで実行
        2. iTerm 等の他ターミナルソフトから「code .」コマンドで VSCode を起動してターミナル経由で実行
            →　起動の際の親プログラムから権限（セキュリティ制約）が引き継がれるらしい