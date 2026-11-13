# 第三方程式來源與授權

## YOLOv5

`src/yolov5/` 包含原專題使用的 YOLOv5 程式及支援模組，上游為 Ultralytics YOLOv5。來源資料夾標示為 `yolov5-6_1`，未記錄確切上游 commit。

保留程式中的來源標頭與隨附的 GPL-3.0 授權全文：[`src/yolov5/LICENSE`](src/yolov5/LICENSE)。

`main.py` 為專題系統入口；`models/`、`utils/`、訓練、評估及匯出工具包含 YOLOv5 程式。自訂資料 YAML 將影像與標註路徑設為儲存庫根目錄的 `dataset/`，類別順序維持 `car`、`motorcycle`。

## 模型與成果圖片

`best_SGD_2612` 模型檔與 `docs/images/` 成品、架構、電路及實驗圖片來自專題資料。程式授權文件不另行宣告這些圖片或模型具有獨立授權條款。
