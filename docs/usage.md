# 執行條件與操作方式

本專案包含 Jetson 端程式與訓練模型。整套系統需搭配雙攝影機、Arduino、馬達與燈具；目前未包含 Arduino 韌體，也沒有完整的原始環境版本鎖定檔。

## 軟體環境

程式使用 Python、PyTorch、Torchvision、OpenCV、CUDA、TensorRT 與 pyserial。JetPack、CUDA、TensorRT、Python 及各套件的確切原始版本未記錄，需在目標 Jetson 上確認相容環境。

[`requirements.txt`](../src/yolov5/requirements.txt) 是隨附 YOLOv5 的最低套件需求，並非能完整還原 Jetson 環境的安裝清單。GPU 軟體與相機環境需先準備完成，再處理一般依賴；不能僅以該檔案判定完整環境已就緒。

主程式另需 pyserial。若其他依賴已備妥，可從儲存庫根目錄安裝這項補充依賴：

```bash
python3 -m pip install -r src/yolov5/requirements-hardware.txt
```

## 模型

| 檔案 | 用途 |
| --- | --- |
| `best_SGD_2612.pt` | PyTorch 模型 |
| `best_SGD_2612.onnx` | ONNX 匯出檔 |
| `best_SGD_2612.engine` | 主程式預設載入的 TensorRT 引擎 |

三者位於 [`src/yolov5/`](../src/yolov5/)，類別順序為 `car`、`motorcycle`。原引擎的建置環境未記錄，需確認能否在目標設備載入。

隨附的 `export.py` 提供 TensorRT 匯出功能。以下是依該程式介面整理的指令，尚未在本次環境實機驗證；匯出會寫入同名 `.engine`，請先保留原檔副本：

```bash
cd src/yolov5
python3 export.py --weights best_SGD_2612.pt --include engine --device 0 --half --data coco128_my.yaml
```

## 啟動前的設定

請對照 [`main.py`](../src/yolov5/main.py) 及實際設備：

| 項目 | 程式設定 | 確認事項 |
| --- | --- | --- |
| 攝影機 | V4L2 裝置 `1`、`2` | 左右順序及實際可讀取的裝置 |
| 影像方向 | `cv2.flip(..., -1)` | 與攝影機安裝方向一致 |
| 影像幾何 | 640 × 480、水平視角假設 60 度 | 實際解析度、視角及校正 |
| 序列埠 | `/dev/ttyS0` | 實體 UART、設定與存取權限 |
| 鮑率 | `115200` | 與 Arduino 一致 |
| 攝影機間距 | `base = 30` cm | 與實際安裝尺寸一致 |
| 燈具高度 | `height_from_ground = 0.86` m | 與實際機構一致 |
| 角度偏移 | `light_angel_bias = 0` | 馬達方向與零點 |
| 控制類別 | `class_num == 0` | 目前僅汽車參與測距及控制 |

## 執行

硬體、控制器韌體與環境備妥後，從儲存庫根目錄執行：

```bash
cd src/yolov5
python3 main.py
```

如果已在 `src/yolov5`，直接執行第二行即可。模型及 YAML 使用相對路徑，需維持這個工作目錄。

- 顯示視窗包含左右辨識畫面、估算距離與處理幀率。
- 主程式會開啟序列埠並送出控制資料，不提供離線模擬控制器模式。
- 在顯示視窗按 `q` 結束。
- 合併影片寫入 `src/yolov5/output.mp4`，再次執行會使用相同檔名。

## 設定檔與支援模組

YAML 是方便閱讀與修改的設定檔格式，`.yaml` 與 `.yml` 都可用來表示此格式。此專案的 YAML 主要分成資料集、模型架構及訓練參數三種；它們本身不包含訓練影像或已訓練的權重。

| 位置 | 內容與用途 |
| --- | --- |
| `coco128_my.yaml` | 專題資料集設定：影像路徑、類別數 `nc: 2` 與類別名稱 `car`、`motorcycle`；主程式從此讀取類別名稱 |
| `yolov5n_my.yaml` | 專題的 YOLOv5n 網路架構設定，指定 2 個類別，以及各層、通道比例與 anchors |
| `models/yolov5s.yaml` | YOLOv5 支援程式的預設架構；專題的 2 類 YOLOv5n 架構另由 `yolov5n_my.yaml` 定義 |
| `data/hyps/hyp.scratch-low.yaml` | `train.py` 的預設訓練參數，包含學習率、動量、損失權重與影像增強強度；不代表原實驗確切採用的設定 |
| `utils/` | 共用工具：影像縮放與增強、資料載入、NMS、座標轉換、畫框、評估與訓練紀錄等 |
| `models/common.py` | 模型共用元件與 `DetectMultiBackend`，負責載入不同格式的模型並執行推論 |
| `models/yolo.py` | YOLO 網路與偵測層定義，將架構設定組成模型 |
| `models/experimental.py` | 額外網路元件及 PyTorch 模型載入工具 |
| `models/tf.py` | TensorFlow 對應網路元件與格式轉換支援，不是主程式預設的 TensorRT 推論入口 |

`models/` 主要是模型定義與載入程式；已訓練的模型檔是同層的 `best_SGD_2612.pt`、`.onnx`、`.engine`。這三種格式用途不同，需分別保留。

各資料夾的 `__init__.py` 是 Python 套件的初始化入口。即使不同資料夾中的檔案內容皆為空白，也有各自的套件位置，不能當成一般重複副本刪除。

## 訓練與評估

`train.py`、`val.py` 與 `yolov5n_my.yaml` 提供模型研究入口。自訂資料設定集中於 [`src/yolov5/coco128_my.yaml`](../src/yolov5/coco128_my.yaml)，使用儲存庫根目錄的 `dataset/`，類別為 `car`、`motorcycle`：

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

資料集未隨儲存庫附上，需自行備妥相同類別與 YOLO 標註格式的資料，才能進行訓練或評估。原訓練超參數與完整執行紀錄不足，不能據此保證重現同一次訓練結果。

[返回專案首頁](../README.md)
