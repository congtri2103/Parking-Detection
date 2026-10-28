# BÁO CÁO ĐỒ ÁN

## HỆ THỐNG NHẬN DIỆN PHƯƠNG TIỆN VÀ PHÁT HIỆN ĐỖ XE SAI QUY ĐỊNH TỪ CAMERA CCTV

**Tác giả:** Nguyễn Công Trí  
**Thời điểm cập nhật:** 14/10/2026  
**Bài toán:** Object Detection, Multi-Object Tracking và suy luận vi phạm đỗ xe  
**Ngôn ngữ:** Python  
**Framework chính:** PyTorch, Torchvision, OpenCV, Supervision/ByteTrack  
**Phần cứng thử nghiệm:** NVIDIA GeForce RTX 2080 Ti 11 GB

---

## 1. Tóm tắt

Đề tài xây dựng một hệ thống xử lý ảnh và video từ camera CCTV nhằm:

1. phát hiện hai loại phương tiện là `car` và `motorcycle`;
2. xác định vị trí của từng phương tiện bằng bounding box;
3. theo dõi phương tiện qua nhiều frame và gán `track_id`;
4. kiểm tra phương tiện có nằm trong vùng quy định hay không;
5. đánh giá phương tiện đang di chuyển hay đã đứng yên;
6. đo thời gian dừng;
7. kết luận trạng thái đỗ xe và lưu bằng chứng khi có vi phạm.

Detector sử dụng Faster R-CNN. Phần Backbone MobileNetV3-Large và Feature
Pyramid Network được tách thành các module riêng trong dự án. Phiên bản hiện tại
khóa cứng `weights=None`, do đó quá trình train mới không tải trọng số COCO hoặc
ImageNet mà khởi tạo toàn bộ mạng từ đầu. Các thành phần Anchor Generator, RPN,
Multi-Scale RoI Align và Detection Head sử dụng implementation chuẩn của
Torchvision, nhưng cũng được khởi tạo ngẫu nhiên trong lần train mới.

Hệ thống không train lớp `illegal_parking`. Kết luận vi phạm được tạo bởi luật
không gian và thời gian sau bước detection/tracking. Cách thiết kế này cho phép
thay đổi vùng cấm và thời gian cho phép theo từng camera mà không cần train lại
detector.

Thực nghiệm 20 epoch trên tập validation cho kết quả tốt nhất tại epoch 8 với
`mAP@0.50=0.4046` và `val_loss=0.6588`. Tại checkpoint này, Car AP đạt 0.6450,
Motorcycle AP đạt 0.1641. Pipeline video đã xử lý end-to-end, phát hiện sự kiện
vi phạm, lưu ảnh bằng chứng và được tích hợp vào web demo nội bộ.

---

## 2. Mục tiêu và phạm vi

### 2.1. Mục tiêu chức năng

- Nhận diện ô tô và xe máy trong ảnh/video CCTV.
- Xuất bounding box, tên lớp và confidence.
- Theo dõi mỗi xe bằng một ID tạm thời.
- Hỗ trợ polygon vùng cấm đỗ cho video.
- Hỗ trợ polygon vùng được phép đỗ cho demo ảnh tĩnh.
- Phân tích chuyển động dựa trên quỹ đạo điểm giữa cạnh dưới bounding box.
- Đếm thời gian xe đứng yên trong vùng cấm.
- Sinh video đã chú thích, ảnh bằng chứng và CSV log.
- Đánh giá detector bằng AP, mAP@0.50, precision, recall và F1.
- Chạy train tái lập được bằng Docker và NVIDIA GPU.

### 2.2. Phạm vi lớp đối tượng

| ID nội bộ | Lớp | Ý nghĩa |
|---:|---|---|
| 0 | background | nền, không phải phương tiện cần phát hiện |
| 1 | car | ô tô |
| 2 | motorcycle | xe máy/mô tô |

Các category gốc có tên `motorcycle` hoặc `motorbike` đều được ánh xạ về lớp 2.

### 2.3. Ngoài phạm vi hiện tại

- Chưa nhận dạng biển số xe.
- Chưa nhận dạng danh tính người lái.
- Chưa phân biệt đầy đủ bus, truck, bicycle hoặc các lớp giao thông khác.
- Chưa học trực tiếp lớp `illegal_parking` bằng neural network.
- Chưa có ground truth tracking để tính IDF1/HOTA.
- Chưa có ground truth sự kiện vi phạm để tính Violation F1.
- Demo ảnh tĩnh không thể chứng minh xe thực sự đứng yên.

---

## 3. Kiến trúc tổng thể

```text
Camera CCTV / Video / Ảnh
            |
            v
  Tiền xử lý và resize ảnh
            |
            v
MobileNetV3-Large Backbone
      C3 / C4 / C5
            |
            v
 Custom Feature Pyramid Network
      P3 / P4 / P5
            |
            v
 Faster R-CNN: RPN + RoI Align + Head
            |
            v
Bounding box + class + confidence
            |
            v
    ByteTrack (đối với video)
            |
            v
 Track ID + lịch sử bottom-center
            |
            v
 ROI + Motion Analysis + Parking Timer
            |
            v
 MOVING / STOPPED / PARKED / VIOLATION
            |
            v
Ảnh/video kết quả + JSON/CSV + ảnh bằng chứng
```

Hệ thống được chia thành hai tầng:

- **Tầng học máy:** phát hiện `car` và `motorcycle`.
- **Tầng luật nghiệp vụ:** xác định vùng, chuyển động, thời gian dừng và vi phạm.

---

## 4. Cấu trúc dự án

```text
illegal-parking-detection/
├── README.md
├── README_FasterRCNN_Parking.md
├── README_WEB_DEMO.md
├── readme_bao_cao.md
├── requirements.txt
├── requirements-docker.txt
├── config.py
├── Dockerfile
├── docker-compose.yml
├── test_modules.py
├── smoke_test_detector.py
├── web_demo.py
│
├── config/
│   ├── no_parking_roi.example.json
│   ├── demo_allowed_parking_roi.json
│   └── demo_no_parking_roi.json
│
├── data/
│   ├── source/
│   ├── train/images/
│   ├── val/images/
│   ├── test/images/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── input/
│   ├── demo_parking_scene.png
│   └── demo_parking_video.mp4
│
├── models/
│   ├── backbone.py
│   ├── fpn.py
│   ├── detector.py
│   ├── best.pth
│   ├── best_loss.pth
│   └── best.history.json
│
├── scripts/
│   ├── split_coco.py
│   └── bdd100k/
│       ├── download_dataset.py
│       └── build_dataset.py
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── detect.py
│   ├── detect_parking_image.py
│   ├── tracker.py
│   ├── roi.py
│   ├── motion.py
│   ├── parking_state.py
│   └── pipeline.py
│
├── tests/
│   └── test_rules.py
│
├── web/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
└── output/
    ├── videos/
    ├── violations/
    ├── logs/
    └── web/<job_id>/
```

---

## 5. Dữ liệu

### 5.1. Nguồn dữ liệu

Dataset được xây dựng từ BDD100K, sau đó lọc chỉ giữ annotation của `car` và
`motorcycle`. Dữ liệu dùng định dạng COCO JSON với ba thành phần chính:

```json
{
  "images": [],
  "annotations": [],
  "categories": []
}
```

Bounding box COCO có dạng `[x, y, width, height]`. Dataset loader chuyển sang
dạng Faster R-CNN yêu cầu:

```text
[x1, y1, x2, y2]
```

### 5.2. Quy mô dữ liệu hiện tại

Sau khi lọc, tập dữ liệu có 9.883 ảnh hợp lệ.

| Split | Số ảnh | Car boxes | Motorcycle boxes | Tổng boxes |
|---|---:|---:|---:|---:|
| Train | 7.905 | 82.294 | 369 | 82.663 |
| Validation | 989 | 10.304 | 46 | 10.350 |
| Test | 989 | 10.239 | 45 | 10.284 |
| **Tổng** | **9.883** | **102.837** | **460** | **103.297** |

Số ảnh có chứa motorcycle:

| Split | Ảnh có motorcycle |
|---|---:|
| Train | 276 |
| Validation | 35 |
| Test | 35 |

Dữ liệu lệch lớp rất mạnh: số motorcycle box nhỏ hơn car box nhiều lần. Đây là
nguyên nhân chính khiến AP của motorcycle thấp và dao động mạnh.

### 5.3. Chiến lược chia dữ liệu

Script `scripts/split_coco.py` thực hiện:

1. chia ảnh thành bucket có motorcycle và bucket car-only;
2. xáo trộn từng bucket với `seed=42`;
3. chia từng bucket theo tỷ lệ xấp xỉ 80% train, 10% validation, 10% test;
4. kiểm tra một `image_id` không xuất hiện ở nhiều split;
5. tạo thư mục ảnh bằng hard link để không nhân bản dung lượng.

Lệnh tái tạo split:

```bash
python3 scripts/split_coco.py \
  --annotations data/source/annotations/instances.json \
  --images-dir data/source/images \
  --output-dir data/annotations \
  --val-ratio 0.10 \
  --test-ratio 0.10 \
  --materialize hardlink \
  --seed 42
```

### 5.4. Hạn chế của cách chia hiện tại

Script bảo đảm không trùng ảnh nhưng chưa nhóm theo video/camera. Nếu hai frame
liên tiếp của cùng một video bị đưa vào hai split khác nhau, metric có thể cao
hơn thực tế do data leakage. Phiên bản triển khai thực tế nên chia theo camera,
video hoặc sequence ID trước, sau đó mới cân bằng lớp trong từng split.

### 5.5. Cân bằng lớp khi train

Train loader dùng `WeightedRandomSampler`. Ảnh có motorcycle nhận trọng số mặc
định 8.0, ảnh car-only nhận trọng số 1.0:

```text
Ảnh car-only       → sampling weight = 1.0
Ảnh có motorcycle → sampling weight = 8.0
```

Kỹ thuật này tăng xác suất chọn ảnh có motorcycle trong mỗi epoch nhưng không
sao chép file ảnh trên ổ đĩa. Validation và test không sử dụng weighted sampling.

---

## 6. Tiền xử lý dữ liệu

`CocoVehicleDataset` thực hiện:

1. mở ảnh bằng Pillow và chuyển về RGB;
2. chuyển ảnh sang tensor float32 trong khoảng `[0, 1]`;
3. lọc annotation không thuộc hai lớp hỗ trợ;
4. loại bounding box có width hoặc height không hợp lệ;
5. trả target gồm `boxes`, `labels`, `image_id`, `area`, `iscrowd`.

Faster R-CNN transform tiếp tục resize ảnh sao cho:

| Tham số | Giá trị |
|---|---:|
| `min_size` | 800 pixel |
| `max_size` | 1333 pixel |

Các ảnh trong cùng batch có thể khác kích thước; GeneralizedRCNNTransform sẽ
padding để tạo batch tensor phù hợp.

---

## 7. Backbone MobileNetV3-Large tự xây dựng

### 7.1. Khởi tạo trọng số

Backbone được tạo bằng:

```python
mobilenet_v3_large(weights=None)
```

Không tồn tại tùy chọn `Weights.DEFAULT` hoặc cờ pretrained trong luồng train.
Vì vậy không có tải trọng số ImageNet/COCO ngầm. Trọng số được khởi tạo ngẫu
nhiên theo cơ chế khởi tạo của PyTorch/Torchvision.

### 7.2. Feature maps được chọn

Backbone lấy output tại các module 6, 12 và 15:

```python
self.return_layers = {6: "C3", 12: "C4", 15: "C5"}
```

Với tensor test `[1, 3, 256, 384]`:

| Feature | Shape | Số kênh | Stride |
|---|---|---:|---:|
| C3 | `[1, 40, 32, 48]` | 40 | 8 |
| C4 | `[1, 112, 16, 24]` | 112 | 16 |
| C5 | `[1, 160, 8, 12]` | 160 | 32 |

C3 có độ phân giải lớn hơn, hữu ích cho xe nhỏ. C5 có ngữ nghĩa mạnh hơn, hữu
ích cho xe lớn hoặc cấu trúc phức tạp.

---

## 8. Feature Pyramid Network tự xây dựng

FPN không gọi `torchvision.ops.FeaturePyramidNetwork`. Module nhận C3/C4/C5 và
tạo P3/P4/P5 đều có 256 kênh.

### 8.1. Lateral convolution

Ba convolution 1×1 biến đổi số kênh:

```text
C3:  40 → 256
C4: 112 → 256
C5: 160 → 256
```

### 8.2. Top-down pathway

```text
M5 = L5
M4 = L4 + Upsample(M5)
M3 = L3 + Upsample(M4)
```

Upsample sử dụng nearest neighbor và resize theo kích thước spatial của feature
đích. FPN cộng feature theo phần tử, không concatenate.

### 8.3. Output convolution

Mỗi feature sau top-down đi qua convolution 3×3, padding 1:

| Feature | Shape với input test 256×384 |
|---|---|
| P3 | `[1, 256, 32, 48]` |
| P4 | `[1, 256, 16, 24]` |
| P5 | `[1, 256, 8, 12]` |

Sơ đồ:

```text
C3 [40,H/8,W/8]   → 1×1 → L3 ───────────── (+) → 3×3 → P3
                                           ↑
C4 [112,H/16,W/16]→ 1×1 → L4 ────── (+) ──┘    → 3×3 → P4
                                    ↑
C5 [160,H/32,W/32]→ 1×1 → L5 ──────┘             → 3×3 → P5
```

---

## 9. Faster R-CNN Detector

### 9.1. Kết nối Backbone/FPN

`MobileNetV3WithFPN` có `out_channels=256` và trả về `OrderedDict` chứa P3, P4,
P5. Đây là giao diện Backbone mà `torchvision.models.detection.FasterRCNN` yêu
cầu.

### 9.2. Anchor Generator

| Feature | Anchor sizes |
|---|---|
| P3 | 32, 64 |
| P4 | 128, 256 |
| P5 | 256, 512 |

Aspect ratios trên mỗi mức:

```text
0.5, 1.0, 2.0
```

Mỗi vị trí feature có 2 kích thước × 3 aspect ratio = 6 anchors.

### 9.3. Region Proposal Network

RPN dự đoán:

- objectness: anchor có chứa đối tượng hay không;
- bounding-box regression: độ lệch để biến anchor thành proposal.

RPN và matching/sampling sử dụng implementation chuẩn của Torchvision. Trọng số
RPN được khởi tạo mới trong train scratch.

### 9.4. Multi-Scale RoI Align

```text
Feature maps    : P3, P4, P5
Output size     : 7×7
Sampling ratio  : 2
```

RoI Align lấy feature có kích thước cố định cho từng proposal trước khi đưa vào
Detection Head.

### 9.5. Detection Head

Head phân loại proposal thành background/car/motorcycle và hồi quy bounding
box. Số lớp gồm background:

```text
num_classes = 3
```

### 9.6. Tương thích checkpoint cũ

Checkpoint mới lưu:

```json
{
  "architecture": "custom_mobilenet_v3_large_fpn_v1",
  "initialization": "scratch"
}
```

Checkpoint legacy không có trường `architecture` được mở bằng kiến trúc
`torchvision_mobilenet_v3_large_fpn`, khởi tạo với `weights=None`, sau đó nạp
state dict. Nhánh legacy chỉ phục vụ inference checkpoint cũ; train mới luôn gọi
custom detector.

---

## 10. Hàm loss và tối ưu

Tổng loss của Faster R-CNN:

```text
L_total = L_RPN_objectness
        + L_RPN_box_regression
        + L_ROI_classification
        + L_ROI_box_regression
```

### 10.1. Tham số train

| Nhóm | Tham số | Giá trị |
|---|---|---:|
| Architecture | identifier | `custom_mobilenet_v3_large_fpn_v1` |
| Initialization | weights | scratch, `weights=None` |
| Epoch | `epochs` | 20 |
| Batch size | Docker mặc định | 2 |
| Batch size | khi GPU thiếu VRAM | 1 |
| Workers | Docker mặc định | 4 |
| Optimizer | loại | SGD |
| Learning rate | ban đầu | 0.005 |
| Momentum | | 0.9 |
| Weight decay | | 0.0005 |
| Scheduler | | StepLR |
| Step size | | 5 epoch |
| Gamma | | 0.1 |
| AMP | CUDA | bật |
| Motorcycle sampling weight | | 8.0 |
| Metric score threshold | | 0.05 |
| Metric IoU threshold | | 0.50 |

Learning-rate schedule:

```text
Epoch 1–5   : 5×10^-3
Epoch 6–10  : 5×10^-4
Epoch 11–15 : 5×10^-5
Epoch 16–20 : 5×10^-6
```

### 10.2. Automatic Mixed Precision

AMP sử dụng:

```python
torch.autocast(device_type="cuda")
torch.amp.GradScaler("cuda")
```

AMP giảm VRAM và tăng tốc trên GPU có Tensor Core. Khi GPU thiếu VRAM, ưu tiên
giảm batch size trước khi giảm kích thước ảnh.

### 10.3. Validation loss và BatchNorm

Torchvision detection model chỉ trả loss khi model ở train mode. Vì vậy, khi
tính validation loss, model vẫn ở train mode nhưng toàn bộ BatchNorm được chuyển
sang eval mode. Cách này ngăn ảnh validation cập nhật running mean/variance của
custom MobileNet backbone và tránh rò rỉ thông tin validation.

### 10.4. Checkpoint

Sau mỗi epoch:

- `models/best.pth`: checkpoint có mAP@0.50 cao nhất;
- `models/best_loss.pth`: checkpoint có validation loss thấp nhất;
- `models/best.history.json`: lịch sử được cập nhật ngay sau từng epoch.

Checkpoint chứa model state, architecture, initialization, epoch, val loss,
metrics, selection metric và class names.

---

## 11. Các chỉ số đánh giá

### 11.1. Intersection over Union

```text
IoU = Area(Prediction ∩ GroundTruth) / Area(Prediction ∪ GroundTruth)
```

Prediction được coi là True Positive khi IoU đạt ngưỡng và ground-truth box đó
chưa được ghép với prediction confidence cao hơn.

### 11.2. Precision, Recall và F1

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Metric tổng hợp là macro-average của `car` và `motorcycle`.

### 11.3. AP và mAP

AP sử dụng precision nội suy tại 101 mức recall. Code hiện đánh giá tại một
ngưỡng IoU duy nhất, mặc định 0.50:

```text
mAP@0.50 = mean(AP_car@0.50, AP_motorcycle@0.50)
```

Đây không phải COCO mAP@[0.50:0.95]. COCO mAP chuẩn phải lấy trung bình AP tại
10 ngưỡng IoU từ 0.50 đến 0.95, bước 0.05.

Score threshold 0.05 được dùng để không cắt mất phần lớn đường
precision-recall. Vì giữ nhiều prediction confidence thấp, precision cuối đường
có thể thấp. Khi demo thực tế thường dùng confidence 0.50.

---

## 12. Kết quả thực nghiệm đã có

### 12.1. Phạm vi kết quả

Kết quả 20 epoch dưới đây thuộc mô hình chính thức
`custom_mobilenet_v3_large_fpn_v1`. Checkpoint xác nhận:

```text
Architecture   : custom_mobilenet_v3_large_fpn_v1
Initialization : scratch
Selection      : mAP@IoU
```

Toàn bộ Backbone, FPN, RPN và RoI Head được khởi tạo ngẫu nhiên; không sử dụng
trọng số ImageNet hoặc COCO. Các metric được tính trên tập validation với
`IoU=0.50` và score threshold `0.05`. Vì vậy `mAP` trong bảng là **mAP@0.50 tại
một ngưỡng IoU**, không phải COCO mAP@[0.50:0.95].

### 12.2. Lịch sử train custom scratch 20 epoch

| Epoch | Train loss | Val loss | LR | mAP@0.50 | Precision | Recall | F1 | Car AP | Motorcycle AP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8578 | 0.8470 | 5e-3 | 0.2313 | 0.0475 | 0.4621 | 0.0847 | 0.4420 | 0.0205 |
| 2 | 0.6944 | 0.8548 | 5e-3 | 0.2873 | 0.0767 | 0.5838 | 0.1297 | 0.5096 | 0.0651 |
| 3 | 0.6230 | 0.7177 | 5e-3 | 0.3157 | 0.0646 | **0.6220** | 0.1132 | 0.5611 | 0.0703 |
| 4 | 0.5934 | 0.7466 | 5e-3 | 0.3021 | 0.0834 | 0.4835 | 0.1421 | 0.5526 | 0.0517 |
| 5 | 0.5835 | 0.7389 | 5e-3 | 0.3402 | 0.0891 | 0.5989 | 0.1513 | 0.5825 | 0.0980 |
| 6 | 0.4980 | 0.6772 | 5e-4 | 0.3874 | 0.1019 | 0.5953 | 0.1710 | 0.6355 | 0.1393 |
| 7 | 0.4761 | 0.6817 | 5e-4 | 0.3930 | 0.1114 | 0.5993 | 0.1836 | 0.6348 | 0.1512 |
| 8 | 0.4630 | **0.6588** | 5e-4 | **0.4046** | 0.1010 | 0.5641 | 0.1699 | 0.6450 | **0.1641** |
| 9 | 0.4532 | 0.6887 | 5e-4 | 0.3929 | 0.1160 | 0.5684 | 0.1896 | 0.6383 | 0.1476 |
| 10 | 0.4481 | 0.6871 | 5e-4 | 0.3774 | 0.1169 | 0.5526 | 0.1903 | 0.6311 | 0.1236 |
| 11 | 0.4308 | 0.6706 | 5e-5 | 0.3848 | 0.1183 | 0.5605 | 0.1943 | 0.6454 | 0.1242 |
| 12 | 0.4219 | 0.6680 | 5e-5 | 0.3784 | 0.1179 | 0.5599 | 0.1935 | **0.6479** | 0.1089 |
| 13 | 0.4209 | 0.6770 | 5e-5 | 0.3709 | 0.1212 | 0.5462 | 0.1972 | 0.6415 | 0.1002 |
| 14 | 0.4219 | 0.6632 | 5e-5 | 0.3750 | 0.1188 | 0.5588 | 0.1949 | 0.6470 | 0.1031 |
| 15 | 0.4162 | 0.6734 | 5e-5 | 0.3719 | 0.1166 | 0.5480 | 0.1915 | 0.6429 | 0.1008 |
| 16 | **0.4106** | 0.6650 | 5e-6 | 0.3762 | 0.1195 | 0.5480 | 0.1951 | 0.6467 | 0.1057 |
| 17 | 0.4154 | 0.6803 | 5e-6 | 0.3722 | **0.1246** | 0.5556 | **0.2026** | 0.6406 | 0.1037 |
| 18 | 0.4173 | 0.6685 | 5e-6 | 0.3765 | 0.1224 | 0.5586 | 0.1998 | 0.6466 | 0.1064 |
| 19 | 0.4175 | 0.6732 | 5e-6 | 0.3759 | 0.1200 | 0.5466 | 0.1961 | 0.6422 | 0.1096 |
| 20 | 0.4174 | 0.6772 | 5e-6 | 0.3716 | 0.1194 | 0.5570 | 0.1955 | 0.6420 | 0.1013 |

### 12.3. Kết quả tốt nhất và checkpoint được chọn

| Tiêu chí | Epoch | Giá trị |
|---|---:|---:|
| Validation loss thấp nhất | 8 | 0.6588 |
| mAP@0.50 cao nhất | 8 | 0.4046 |
| Car AP cao nhất | 12 | 0.6479 |
| Motorcycle AP cao nhất | 8 | 0.1641 |
| Precision cao nhất tại score 0.05 | 17 | 0.1246 |
| Recall cao nhất tại score 0.05 | 3 | 0.6220 |
| F1 cao nhất tại score 0.05 | 17 | 0.2026 |

Hai tiêu chí chính là mAP và validation loss cùng chọn epoch 8. Vì vậy:

```text
models/best.pth      → epoch 8, chọn theo mAP@0.50
models/best_loss.pth → epoch 8, chọn theo validation loss
```

Checkpoint dùng cho demo và triển khai là `models/best.pth`.

### 12.4. Phân tích quá trình hội tụ

- Trong tám epoch đầu, train loss giảm từ 0.8578 xuống 0.4630, validation loss
  giảm từ 0.8470 xuống 0.6588 và mAP@0.50 tăng từ 0.2313 lên 0.4046. Điều này
  chứng minh mạng khởi tạo ngẫu nhiên đã học được đặc trưng có ích.
- StepLR giảm learning rate sau mỗi năm epoch: `5e-3 → 5e-4 → 5e-5 → 5e-6`.
  Bước giảm đầu tiên tạo ra cải thiện rõ nhất ở epoch 6–8.
- Sau epoch 8, train loss tiếp tục giảm tới 0.4106 nhưng validation loss không
  thấp hơn 0.6588 và mAP dao động khoảng 0.37–0.39. Đây là dấu hiệu bão hòa và
  overfitting nhẹ.
- Epoch 20 không phải checkpoint tốt nhất. Sử dụng epoch cuối sẽ làm mAP giảm từ
  0.4046 xuống 0.3716.
- Với run tương tự, early stopping khoảng epoch 8–12 là hợp lý hơn tiếp tục train
  cố định 20 epoch.

### 12.5. Phân tích theo lớp

Tại checkpoint epoch 8:

| Lớp | Ground truth | Predictions tại score ≥ 0.05 | Precision | Recall | F1 | AP@0.50 |
|---|---:|---:|---:|---:|---:|---:|
| Car | 10.304 | 49.800 | 0.1660 | 0.8021 | 0.2750 | 0.6450 |
| Motorcycle | 46 | 417 | 0.0360 | 0.3261 | 0.0648 | 0.1641 |

Car AP đạt mức 0.6450 tại checkpoint tốt nhất, trong khi Motorcycle AP chỉ đạt
0.1641. Khoảng cách này phù hợp với mất cân bằng dữ liệu: tập train có 82.294
car box nhưng chỉ có 369 motorcycle box, tương đương xấp xỉ 223:1. Weighted
sampling làm tăng tần suất ảnh motorcycle xuất hiện nhưng không tạo thêm độ đa
dạng về góc nhìn, kích thước, che khuất hoặc điều kiện ánh sáng.

### 12.6. Giải thích precision thấp

Precision, recall và F1 trong bảng được lấy tại score threshold 0.05. Ngưỡng
này cố ý thấp để giữ gần đầy đủ đường precision-recall khi tính AP, nhưng đồng
thời giữ lại rất nhiều box confidence thấp. Vì vậy precision 0.1010 tại epoch 8
không đồng nghĩa web demo ở confidence 0.30 hoặc 0.50 cũng có precision 10,10%.

Để xác định operating point triển khai, cần chạy validation tại nhiều threshold,
ví dụ 0.15, 0.30, 0.50 và 0.70, rồi chọn theo yêu cầu nghiệp vụ giữa bỏ sót và
cảnh báo sai.

### 12.7. Giới hạn của kết quả

- Đây là kết quả validation, chưa phải đánh giá cuối cùng trên test set.
- AP sử dụng nội suy 101 điểm tại một ngưỡng IoU 0.50; chưa phải COCO mAP đầy đủ.
- Validation chỉ có 46 motorcycle box nên metric lớp này có phương sai lớn.
- Dataset detection chưa có ground truth sự kiện đỗ sai, vì vậy mAP detector
  không đo trực tiếp chất lượng kết luận `VIOLATION`.

---

## 13. Inference trên ảnh

### 13.1. Phát hiện phương tiện đơn thuần

```bash
python -m src.detect input/test.jpg \
  --checkpoint models/best.pth \
  --confidence 0.50 \
  --output output/detection.jpg
```

Kết quả gồm bounding box, lớp và confidence.

### 13.2. Demo đỗ đúng/sai bằng ảnh tĩnh

`src.detect_parking_image` nhận polygon vùng được phép đỗ. Với mỗi box, hệ thống
lấy điểm giữa cạnh dưới:

```text
bottom_center = ((x1 + x2) / 2, y2)
```

Quy tắc:

```text
bottom-center nằm trong allowed ROI  → PARKING LEGAL
bottom-center nằm ngoài allowed ROI → PARKING ILLEGAL
```

Kết quả trực quan:

- khung xanh: `PARKING LEGAL`;
- khung đỏ: `PARKING ILLEGAL`;
- polygon vùng được phép đỗ;
- class và confidence;
- tổng số xe đúng/sai;
- JSON chi tiết từng detection.

Ảnh demo có kích thước 1672×941. ROI mẫu:

```json
{
  "points": [[400, 80], [1671, 200], [1671, 555], [390, 246]]
}
```

Lệnh chạy:

```bash
python -m src.detect_parking_image input/demo_parking_scene.png \
  --allowed-roi-file config/demo_allowed_parking_roi.json \
  --checkpoint models/best.pth \
  --confidence 0.50 \
  --output output/demo_parking_result.jpg
```

**Giới hạn:** ảnh tĩnh chỉ đánh giá quan hệ không gian, không biết xe đang chạy,
dừng đèn đỏ hay thực sự đỗ. Kết luận vi phạm đáng tin cậy cần chuỗi video.

---

## 14. Pipeline video và ByteTrack

### 14.1. Tracking

`ByteTrackAdapter` chuyển detection thành `supervision.Detections`, sau đó gọi
ByteTrack để trả về:

```text
track_id, bounding box, confidence, class_id
```

Tham số mặc định:

| Tham số | Giá trị |
|---|---:|
| Track activation threshold | 0.25 |
| Lost track buffer | 30 frame |
| Minimum matching threshold | 0.80 |
| Frame rate fallback | 30 FPS |

Track ID chỉ là ID tạm trong video, không phải biển số xe.

### 14.2. ROI

Pipeline video dùng polygon `NO PARKING`. Điểm bottom-center của xe được kiểm tra
bằng ray casting. Điểm trên cạnh polygon được xem là nằm trong vùng.

### 14.3. Phân tích chuyển động

Với mỗi track, hệ thống lưu lịch sử:

```text
(timestamp, bottom_center)
```

Trong cửa sổ xác nhận, displacement là khoảng cách lớn nhất từ điểm neo đến các
điểm gần đây. Xe được xem là đứng yên khi:

```text
observed_seconds >= confirmation_seconds
AND displacement_px <= motion_threshold
```

Tham số mặc định:

| Tham số | Giá trị |
|---|---:|
| Motion threshold | 8 pixel |
| Confirmation time | 2 giây |
| History time | 4 giây |

### 14.4. State machine đỗ xe

```text
Xe chuyển động
    → MOVING

Xe đứng yên ngoài vùng cấm
    → STOPPED, không vi phạm

Xe đứng yên trong vùng cấm nhưng chưa đủ thời gian
    → STOPPED

Xe đứng yên trong vùng cấm đủ parking_seconds
    → PARKED + VIOLATION
```

Tham số:

| Tham số | Giá trị |
|---|---:|
| Parking time threshold | 10 giây |
| Lost track expiry | 3 giây |

Khi vi phạm bắt đầu, hệ thống chỉ phát một sự kiện `violation_started` và lưu
một ảnh bằng chứng. Khi xe di chuyển trở lại hoặc rời vùng, bộ đếm được reset.

### 14.5. Lệnh chạy video

```bash
python -m src.pipeline input/test.mp4 \
  --roi-file config/no_parking_roi.json \
  --checkpoint models/best.pth \
  --confidence 0.50 \
  --parking-seconds 10 \
  --motion-threshold 8 \
  --device cuda
```

Output:

```text
output/videos/result.mp4
output/logs/result.csv
output/violations/violation_ID...jpg
```

CSV gồm thời gian video, frame, track ID, class, confidence, zone, displacement,
thời gian dừng, state và cờ violation.

### 14.6. Web demo

Web demo nội bộ được triển khai bằng `web_demo.py` và giao diện HTML/CSS/JS
trong thư mục `web/`. Web không gửi video hoặc checkpoint ra dịch vụ bên ngoài.
Các chức năng chính gồm:

- dùng video mẫu hoặc tải video riêng tối đa 512 MB;
- vẽ polygon vùng cấm trực tiếp trên khung hình;
- cấu hình confidence, thời gian đỗ và ngưỡng chuyển động;
- theo dõi log và tiến độ xử lý;
- xem thống kê track, vi phạm và thời gian dừng;
- phát video kết quả H.264, tải CSV và xem ảnh bằng chứng.

Mỗi job được lưu tách biệt:

```text
output/web/<job_id>/
├── videos/result_raw.mp4
├── videos/result.mp4
├── logs/result_raw.csv
└── violations/violation_ID*.jpg
```

Lệnh chạy an toàn bằng CPU:

```bash
WEB_DEVICE=cpu WEB_PORT=8501 docker compose up -d web
```

Demo video 20 giây đã xử lý đủ 300 frame, theo dõi được xe `track_id=6`, chuyển
trạng thái sang `PARKED`, đặt `violation=1` và lưu ảnh bằng chứng. Kết quả này
xác nhận pipeline kỹ thuật hoạt động end-to-end, nhưng video mô phỏng từ một ảnh
tĩnh chưa thay thế được đánh giá trên video CCTV thực tế.

---

## 15. Docker và môi trường GPU

### 15.1. Docker image

Base image mặc định:

```text
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

Dockerfile cài FFmpeg, OpenCV headless, NumPy, Pillow, Supervision, tqdm và
pytest. Project được copy vào `/workspace`.

### 15.2. Volume

| Host | Container | Chế độ |
|---|---|---|
| `./data` | `/workspace/data` | read-only |
| `./input` | `/workspace/input` | read-only |
| `./models` | `/workspace/models` | read-write |
| `./output` | `/workspace/output` | read-write |
| torch cache | `/root/.cache/torch` | read-write |

### 15.3. Chọn GPU

Compose mới dùng:

```yaml
device_ids: ["${GPU_ID:-0}"]
```

Điều này giới hạn container vào đúng GPU vật lý được chọn. Với image/Compose cũ
có thể ép thêm GPU logic số 0 sau khi Docker đã ánh xạ GPU vật lý:

```bash
-e CUDA_VISIBLE_DEVICES=0
```

Sau khi remap, GPU vật lý 1 xuất hiện trong container dưới tên logic `cuda:0`.

Máy thử nghiệm có ba RTX 2080 Ti, mỗi GPU khoảng 11 GB VRAM. Cần chạy
`nvidia-smi` trước khi train, không chọn GPU chỉ dựa vào utilization vì GPU có
thể utilization 0% nhưng VRAM vẫn bị process khác giữ.

### 15.4. Lỗi build đã gặp

Docker từng không tải được metadata base image do:

```text
TLS handshake failure với registry-1.docker.io
```

Đây là lỗi mạng/registry, không phải lỗi source. Khi image cũ đã có đủ thư viện,
có thể mount source mới tạm thời:

```bash
docker compose run --rm \
  -v "$PWD:/workspace" \
  train ...
```

Khi mạng ổn định cần build lại để image chứa source mới cố định.

---

## 16. Quy trình kiểm thử

### 16.1. Unit test luật nghiệp vụ

```bash
python -m unittest discover -v
```

Các test bao phủ:

- bottom-center;
- point-in-polygon và điểm nằm trên biên;
- parse ROI;
- xác nhận đứng yên;
- reset lịch sử khi timestamp không tăng;
- timestamp frame không đều;
- sự kiện vi phạm chỉ phát một lần;
- xe đứng yên ngoài vùng cấm không bị kết luận vi phạm.

### 16.2. Shape test Backbone/FPN

```bash
python test_modules.py
```

Kết quả mong đợi:

```text
C3 (1, 40, 32, 48)
C4 (1, 112, 16, 24)
C5 (1, 160, 8, 12)
P3 (1, 256, 32, 48)
P4 (1, 256, 16, 24)
P5 (1, 256, 8, 12)
Detector backbone: custom MobileNetV3WithFPN (scratch)
PASS: backbone and FPN shape tests
```

### 16.3. Smoke test detector

Smoke test chạy một forward/backward thật trước khi train dài:

```bash
python smoke_test_detector.py \
  --images data/train/images \
  --annotations data/annotations/train.json \
  --device cuda \
  --batch-size 1 \
  --amp
```

Kết quả bắt buộc:

```text
Architecture: custom_mobilenet_v3_large_fpn_v1
Initialization: scratch
PASS: custom detector forward/backward
```

Không bắt đầu train dài nếu smoke test chưa PASS.

---

## 17. Quy trình train đề xuất

### 17.1. Kiểm tra GPU

```bash
nvidia-smi
```

### 17.2. Kiểm tra container nhìn đúng GPU

```bash
GPU_ID=1 docker compose run --rm --no-deps \
  -e CUDA_VISIBLE_DEVICES=0 \
  train \
  python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

### 17.3. Train bằng tmux

```bash
mkdir -p output/logs
LOG_FILE="output/logs/train_scratch_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s cctv-scratch \
  "cd '/home/minhpt36/illegal-parking-detection' && GPU_ID=1 EPOCHS=20 BATCH_SIZE=1 WORKERS=2 LEARNING_RATE=0.005 MOTORCYCLE_WEIGHT=8.0 docker compose run --rm --no-deps -e CUDA_VISIBLE_DEVICES=0 -v '/home/minhpt36/illegal-parking-detection:/workspace' train 2>&1 | tee -a '$LOG_FILE'"
```

Theo dõi:

```bash
tmux attach -t cctv-scratch
tail -f "$(ls -t output/logs/train_scratch_*.log | head -1)"
watch -n 2 nvidia-smi
```

Detach tmux mà không dừng train: nhấn `Ctrl+B`, sau đó `D`.

### 17.4. Xác minh train scratch

Dòng đầu log phải chứa:

```text
architecture=custom_mobilenet_v3_large_fpn_v1
initialization=scratch (weights=None)
```

Kiểm tra checkpoint:

```bash
python -c "
import torch
c=torch.load('models/best.pth', map_location='cpu', weights_only=True)
print(c['architecture'])
print(c['initialization'])
"
```

---

## 18. Đánh giá trên test set

Sau khi chọn model bằng validation, đánh giá một lần trên test:

```bash
python -m src.evaluate \
  --images data/test/images \
  --annotations data/annotations/test.json \
  --checkpoint models/best.pth \
  --score-threshold 0.05 \
  --iou-threshold 0.50 \
  --device cuda \
  --output output/test_evaluation.json
```

Nên chạy thêm ở confidence 0.50 để quan sát precision/recall tại operating point
demo, nhưng AP chuẩn của implementation này nên giữ score threshold đủ thấp.

Không sử dụng test set để chọn epoch, learning rate hoặc confidence threshold;
các quyết định đó phải dựa trên validation set.

---

## 19. Phân tích rủi ro và hạn chế

### 19.1. Train từ đầu

Train scratch trên 9.883 ảnh có thể hội tụ chậm và cho kết quả thấp hơn transfer
learning. 20 epoch có thể chưa đủ. Cần theo dõi mAP validation và điều chỉnh số
epoch/scheduler bằng thực nghiệm, không chỉ nhìn train loss.

### 19.2. Lệch lớp

Motorcycle chỉ có 460 box toàn dataset. Weighted sampling giúp tăng số lần nhìn
thấy ảnh motorcycle nhưng không tạo thêm đa dạng thật. Cần thu thập thêm xe máy,
đặc biệt trong góc camera CCTV tương tự môi trường triển khai.

### 19.3. Domain shift

BDD100K chủ yếu là cảnh giao thông đường phố. Camera cố định tại bãi xe có góc
nhìn, độ cao, độ nén, ánh sáng và mật độ phương tiện khác. Cần fine-tune bằng dữ
liệu đúng camera đích.

### 19.4. ROI thủ công

ROI phải hiệu chỉnh cho từng camera. Camera rung, đổi góc hoặc zoom sẽ làm
polygon sai. Hệ thống nên lưu cấu hình ROI theo camera ID và có công cụ hiệu
chỉnh trực quan.

### 19.5. Bottom-center heuristic

Bottom-center là xấp xỉ điểm tiếp xúc mặt đất. Bounding box bị che khuất hoặc
perspective mạnh có thể làm điểm này nằm sai vùng. Có thể cải thiện bằng
segmentation, homography hoặc hiệu chuẩn mặt phẳng mặt đường.

### 19.6. Ảnh tĩnh

Ảnh tĩnh không phân biệt xe đỗ với xe đang chạy qua vùng. Chế độ ảnh chỉ phù hợp
minh họa spatial rule. Quyết định nghiệp vụ cần video và timer.

### 19.7. Tracking

ByteTrack có thể đổi ID khi xe bị che khuất lâu, làm reset bộ đếm. Cần đánh giá
ID switch trên video thực tế và cân chỉnh buffer/matching threshold.

### 19.8. Metric

mAP hiện tại là mAP@0.50 một ngưỡng IoU, chưa phải COCO mAP@[0.50:0.95]. Báo
cáo kết quả phải ghi đầy đủ tên metric để tránh so sánh sai với công trình khác.

---

## 20. Hướng phát triển

1. Đánh giá checkpoint epoch 8 một lần trên test set và khóa kết quả cuối cùng.
2. Cài COCO evaluator chuẩn để tính AP@[0.50:0.95], AP50, AP75 và AP theo kích thước.
3. Quét confidence threshold trên validation để chọn operating point triển khai.
4. Chia dataset theo video/camera thay vì chỉ theo ảnh.
5. Thu thập thêm motorcycle và dữ liệu camera đích để giảm mất cân bằng 223:1.
6. Bổ sung augmentation phù hợp CCTV: sáng/tối, mưa, blur, compression và scale.
7. Tuning anchor bằng thống kê kích thước box trong dataset.
8. Thêm early stopping theo mAP validation với patience khoảng 3–5 epoch.
9. Lưu checkpoint latest gồm optimizer, scheduler và GradScaler để resume đầy đủ.
10. Mở rộng web demo cho nhiều camera, nhiều ROI, xác thực và hàng đợi job.
11. Thêm perspective calibration hoặc bird's-eye view.
12. Đánh giá tracking bằng IDF1/HOTA khi có ground truth.
13. Xây dựng ground truth sự kiện để đánh giá Violation Precision/Recall/F1.
14. Thử nghiệm transfer learning như một baseline riêng để so sánh công bằng.
15. Tối ưu inference bằng TorchScript/ONNX/TensorRT nếu triển khai realtime.

---

## 21. Kết luận

Đồ án đã xây dựng được một pipeline có cấu trúc rõ ràng từ detection đến suy
luận vi phạm đỗ xe. Điểm chính của hệ thống là tách bài toán thành hai phần:
neural network nhận diện phương tiện và luật không gian-thời gian xác định vi
phạm. Cách tiếp cận này phù hợp với camera cố định vì ROI và thời gian cho phép
có thể cấu hình độc lập.

Phiên bản source hiện tại đã tích hợp Backbone MobileNetV3-Large và FPN tự xây
dựng vào Faster R-CNN, đồng thời khóa `weights=None` để đáp ứng yêu cầu train từ
đầu. Hệ thống có công cụ chuẩn bị dữ liệu, weighted sampling, AMP, metric theo
epoch, checkpoint theo mAP/loss, demo ảnh, pipeline video, Docker và bộ test cho
luật nghiệp vụ.

Mô hình custom train hoàn toàn từ đầu đã hội tụ và đạt mAP@0.50 validation tốt
nhất 0.4046 tại epoch 8. Cùng epoch này, validation loss đạt giá trị thấp nhất
0.6588, Car AP đạt 0.6450 và Motorcycle AP đạt 0.1641. Kết quả chứng minh kiến
trúc tự xây dựng hoạt động đúng, nhưng cũng chỉ ra hai vấn đề chính: overfitting
sau epoch 8 và chất lượng motorcycle thấp do dữ liệu lệch lớp rất mạnh.

Checkpoint `models/best.pth` tại epoch 8 là phiên bản phù hợp để demo. Pipeline
video đã chạy end-to-end, phát hiện một track vi phạm và lưu ảnh bằng chứng; web
demo nội bộ cũng đã được bổ sung để cấu hình ROI và xem kết quả trực quan. Trước
khi tuyên bố hiệu năng triển khai, dự án vẫn cần đánh giá độc lập trên test set,
thu thập video CCTV thực tế và xây dựng ground truth cấp sự kiện vi phạm.

---

## Phụ lục A. Các lệnh thường dùng

### A.1. Kiểm tra không dùng pretrained weights

```bash
grep -R "Weights.DEFAULT" models src
grep -n "weights=None" models/backbone.py models/detector.py
```

Lệnh đầu không được in kết quả. Lệnh thứ hai phải hiển thị `weights=None`.

### A.2. Xem tiến trình

```bash
tmux ls
docker ps --filter label=com.docker.compose.service=train
tail -f "$(ls -t output/logs/train_*.log | head -1)"
watch -n 2 nvidia-smi
```

### A.3. Dừng train an toàn

```bash
tmux attach -t cctv-scratch
```

Nhấn `Ctrl+C`, đợi container dừng, sau đó `exit`.

### A.4. Xem toàn bộ epoch

```bash
python3 -m json.tool models/best.history.json | less
```

### A.5. Kiểm tra checkpoint

```bash
python -c "
import torch
c=torch.load('models/best.pth', map_location='cpu', weights_only=True)
print('Architecture:', c.get('architecture'))
print('Initialization:', c.get('initialization'))
print('Epoch:', c.get('epoch'))
print('Val loss:', c.get('val_loss'))
print('Metrics:', c.get('metrics'))
"
```

---

## Phụ lục B. Quy ước màu hiển thị

| Màu | Ý nghĩa |
|---|---|
| Xanh lá | xe nằm trong vùng được phép ở demo ảnh |
| Đỏ | xe nằm ngoài vùng được phép hoặc vi phạm |
| Vàng/cam | ROI hoặc xe trong vùng cần theo dõi |
| Trắng | điểm bottom-center hoặc thông tin phụ |

---

## Phụ lục C. Phân biệt các loại kết quả

| Kết quả | Ý nghĩa |
|---|---|
| Detection trên ảnh | model thấy phương tiện nào |
| Legal/illegal trên ảnh | box nằm trong/ngoài polygon, chưa xác nhận đứng yên |
| Violation trên video | xe đứng yên trong vùng cấm quá ngưỡng thời gian |
| AP/mAP | chất lượng detector so với ground truth box |
| Violation F1 | chất lượng kết luận sự kiện; hiện chưa đo do thiếu ground truth |
