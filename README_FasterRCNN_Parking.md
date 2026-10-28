# Phát hiện đỗ xe sai quy định từ Camera CCTV

## 1. Mục tiêu

Hệ thống phát hiện **ô tô (car)** và **xe máy (motorcycle)** từ video camera cố định, theo dõi từng phương tiện theo thời gian và kết luận **đỗ sai quy định** khi phương tiện:

1. nằm trong vùng cấm đỗ (No-Parking ROI),
2. ở trạng thái đứng yên,
3. đứng yên liên tục quá ngưỡng `T_PARK`.

> Detector **không học lớp `illegal_parking`**. Trạng thái vi phạm được suy ra bằng Tracking + ROI + Motion + Timer.

---

## 2. Kiến trúc được chốt

```text
Camera / Video
      ↓
Frame Extraction
      ↓
Pre-processing
      ↓
Faster R-CNN + MobileNetV3-Large FPN
      ↓
car / motorcycle
      ↓
Bounding Box + Class + Confidence
      ↓
ByteTrack
      ↓
Track ID + Track History
      ↓
No-Parking ROI
      ↓
Motion Analysis
      ↓
STOPPED ?
      ↓
Stopped Duration > T_PARK ?
      ↓
ILLEGAL PARKING
      ↓
Evidence Image / Video / Log
```

### Vì sao chọn kiến trúc này?

- Faster R-CNN là detector two-stage, phù hợp với hướng nghiên cứu đã chọn.
- Backbone **MobileNetV3-Large + FPN** nhẹ hơn ResNet50-FPN.
- Mô hình Faster R-CNN MobileNetV3-Large FPN chuẩn của Torchvision có khoảng **19.4 triệu tham số**.
- FPN hỗ trợ phát hiện phương tiện ở nhiều kích thước khác nhau trong camera CCTV.
- Chỉ fine-tune cho hai lớp cần thiết: `car` và `motorcycle`.

---

## 3. Kiến trúc Faster R-CNN

```text
Input Frame
    ↓
MobileNetV3-Large Backbone
    ↓
Feature Pyramid Network (FPN)
    ↓
Multi-scale Feature Maps
    ↓
Region Proposal Network (RPN)
    ├── Objectness
    └── Bounding-box Regression
    ↓
Region Proposals
    ↓
RoI Align
    ↓
Detection Head
    ├── Classification
    └── Bounding-box Regression
    ↓
Final Detection
(box, class, confidence)
```

### 3.1 MobileNetV3-Large

Backbone chịu trách nhiệm trích xuất đặc trưng từ frame. MobileNetV3-Large được chọn nhằm giảm chi phí tính toán so với ResNet50 trong khi vẫn giữ kiến trúc Faster R-CNN.

### 3.2 Feature Pyramid Network

FPN tạo đặc trưng ở nhiều scale. Điều này quan trọng với CCTV vì xe ở xa camera thường nhỏ hơn nhiều so với xe ở gần camera.

### 3.3 Region Proposal Network

RPN tìm các vùng có khả năng chứa phương tiện. RPN dự đoán:

- `objectness`: vùng có object hay background;
- `bbox regression`: điều chỉnh anchor thành region proposal.

RPN chưa quyết định đó là car hay motorcycle.

### 3.4 RoI Align

RoI Align lấy đặc trưng của từng proposal từ feature maps và đưa về kích thước cố định để Detection Head xử lý.

### 3.5 Detection Head

Detection Head thực hiện:

- phân loại: `background / car / motorcycle`;
- tinh chỉnh bounding box cuối cùng.

Trong Torchvision, `num_classes` **bao gồm background**, do đó dự án sử dụng:

```python
NUM_CLASSES = 3  # background + car + motorcycle
```

---

## 4. Tham số khởi đầu

| Nhóm | Tham số | Giá trị ban đầu |
|---|---|---:|
| Detector | Model | Faster R-CNN |
| Backbone | Model | MobileNetV3-Large + FPN |
| Model size | Parameters | ~19.4M (mô hình chuẩn trước khi thay head) |
| Classes | Object classes | car, motorcycle |
| Classes | `num_classes` | 3 (gồm background) |
| Input | `min_size` | 800 (high-resolution model) |
| Input | `max_size` | 1333 |
| Detection | score threshold | 0.50 |
| Detection | NMS threshold | 0.50 |
| Optimizer | SGD | Có |
| Learning rate | LR | 0.005 để bắt đầu tuning |
| Momentum | | 0.9 |
| Weight decay | | 0.0005 |
| Batch size | | 2-4 tùy GPU |
| Epochs | | 20-50 tùy dataset |
| Scheduler | | StepLR |
| Tracking | | ByteTrack |
| Motion | `MOTION_THRESHOLD` | hiệu chỉnh trên video |
| Parking | `T_PARK` | hiệu chỉnh theo nghiệp vụ |
| ROI | | Polygon cố định theo camera |

> Các threshold, learning rate, epoch và `T_PARK` là **giá trị khởi đầu/tham số cần tuning**, không phải quy định cố định.

---

## 5. Cấu trúc thư mục đề xuất

```text
illegal-parking-detection/
│
├── README.md
├── README_FasterRCNN_Parking.md
├── requirements.txt
├── requirements-docker.txt
├── config.py
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── train/images/
│   ├── val/images/
│   ├── test/images/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── models/
│   ├── detector.py
│   └── best.pth
│
├── scripts/
│   ├── split_coco.py
│   └── bdd100k/
│       ├── download_dataset.py
│       └── build_dataset.py
│
├── third_party/
│   └── bdd100k/             # mã nguồn/thư viện BDD100K gốc
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── detect.py
│   ├── tracker.py
│   ├── roi.py
│   ├── motion.py
│   ├── parking_state.py
│   └── pipeline.py
│
├── input/
│   └── test.mp4
│
└── output/
    ├── videos/
    ├── violations/
    └── logs/
```

---

## 6. Cài đặt

Khuyến nghị Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate      # Windows

pip install torch torchvision opencv-python numpy pillow matplotlib
```

Cài thêm implementation ByteTrack phù hợp với môi trường dự án trước khi chạy tracking.

---

## 7. Khởi tạo detector

`models/detector.py`

```python
from models.detector import build_custom_detector


def get_model(num_classes=3):
    # Source code thật nối MobileNetV3Backbone + FeaturePyramidNetwork vào
    # FasterRCNN cùng AnchorGenerator và MultiScaleRoIAlign.
    # Backbone khóa cứng mobilenet_v3_large(weights=None).
    return build_custom_detector(num_classes=num_classes)
```

Không có đường code nào tải COCO hoặc ImageNet weights cho lần train mới.
Tham số của Backbone, FPN, RPN, RoI Head và predictor đều bắt đầu từ khởi tạo
ngẫu nhiên. Hàm legacy chỉ được dùng để đọc checkpoint cũ, với
`weights=None` trước khi nạp chính state dict của checkpoint.

---

## 8. Format dữ liệu train

Mỗi ảnh cần annotation bounding box và label.

Ví dụ target của một ảnh:

```python
target = {
    "boxes": tensor([
        [x1, y1, x2, y2],
        [x1, y1, x2, y2],
    ]),
    "labels": tensor([1, 2]),
}
```

Quy ước:

```text
0 = background
1 = car
2 = motorcycle
```

Nên chia dataset theo **video/camera**, không chia ngẫu nhiên các frame liền nhau để tránh data leakage.

---

## 9. Training

Pipeline train:

```text
Dataset
   ↓
DataLoader
   ↓
Faster R-CNN MobileNetV3-FPN
   ↓
Loss
   ├── RPN classification loss
   ├── RPN box regression loss
   ├── RoI classification loss
   └── RoI box regression loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
best.pth
```

Loss tổng:

```text
L_total = L_RPN_cls + L_RPN_box + L_ROI_cls + L_ROI_box
```

Ví dụ optimizer:

```python
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005,
)
```

Lưu checkpoint tốt nhất:

```python
torch.save(model.state_dict(), "models/best.pth")
```

---

## 10. Inference

Với mỗi frame:

```text
Frame
 ↓
Faster R-CNN
 ↓
boxes + labels + scores
 ↓
score >= CONF_THRESHOLD
 ↓
car / motorcycle detections
```

Pseudo-code:

```python
model.eval()

with torch.no_grad():
    prediction = model([image_tensor.to(device)])[0]

keep = prediction["scores"] >= CONF_THRESHOLD
boxes = prediction["boxes"][keep]
labels = prediction["labels"][keep]
scores = prediction["scores"][keep]
```

---

## 11. Tracking

Detection của mỗi frame được chuyển sang ByteTrack:

```text
Frame 100 → car → ID 8
Frame 101 → car → ID 8
Frame 102 → car → ID 8
```

Mỗi Track ID lưu:

```python
track_history[track_id] = [
    {"time": t1, "center": p1, "bbox": box1},
    {"time": t2, "center": p2, "bbox": box2},
]
```

Track ID là ID tạm của tracker, **không phải biển số xe**.

---

## 12. No-Parking ROI

Với camera cố định, vùng cấm được khai báo bằng polygon:

```python
NO_PARKING_ROI = [
    (x1, y1),
    (x2, y2),
    (x3, y3),
    (x4, y4),
]
```

Điểm đại diện cho xe là bottom-center của bounding box:

```python
def bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)
```

Sau đó kiểm tra điểm này có nằm trong polygon cấm hay không.

---

## 13. Motion Analysis

Với cùng một Track ID, tính displacement giữa các thời điểm:

```text
d = sqrt((x_t - x_(t-1))² + (y_t - y_(t-1))²)
```

Logic cơ bản:

```text
d > ε
→ MOVING

d <= ε liên tục trong nhiều frame
→ STOPPED
```

Không nên kết luận STOPPED chỉ từ hai frame liên tiếp. Nên dùng cửa sổ nhiều frame hoặc smoothing để giảm ảnh hưởng bbox jitter.

---

## 14. Parking Timer

Khi một xe chuyển sang `STOPPED`:

```python
stopped_start[track_id] = current_time
```

Thời gian dừng:

```python
stopped_duration = current_time - stopped_start[track_id]
```

State machine:

```text
MOVING
  ↓ đứng yên liên tục
STOPPED
  ↓ duration > T_PARK
PARKED
```

Nếu xe di chuyển trở lại trước `T_PARK`, reset timer.

---

## 15. Rule phát hiện đỗ sai

Điều kiện cuối cùng:

```text
VIOLATION =
    InsideNoParkingROI
    AND Stopped
    AND StoppedDuration > T_PARK
```

Pseudo-code:

```python
if inside_no_parking_roi(track):
    if is_stopped(track):
        if stopped_duration(track) > T_PARK:
            violation = True
```

Đây là phần **quyết định đỗ sai**, không phải Faster R-CNN.

---

## 16. Pipeline chính

`src/pipeline.py` thực hiện:

```text
while video.isOpened():
    1. Đọc frame
    2. Faster R-CNN detect car/motorcycle
    3. Lọc confidence
    4. ByteTrack update
    5. Cập nhật track history
    6. Kiểm tra ROI
    7. Phân tích chuyển động
    8. Cập nhật parking timer
    9. Áp dụng violation rule
   10. Vẽ bbox + ID + state
   11. Lưu evidence nếu vi phạm
   12. Ghi output video/log
```

---

## 17. Output mong muốn

Video demo hiển thị:

```text
ID: 8
Class: car
Confidence: 0.94
State: STOPPED
Duration: 13.2 s
Zone: NO PARKING
Status: VIOLATION
```

Khi vi phạm, lưu:

```text
output/violations/
    violation_ID8_20261007_101530.jpg
```

Log có thể gồm:

```text
timestamp, track_id, class, confidence, zone, stopped_duration, violation
```

---

## 18. Đánh giá

### Detector

- Precision
- Recall
- F1-score
- mAP
- FPS / latency

Đặc biệt ưu tiên **Recall**, vì bỏ sót xe có thể làm mất track và làm hệ thống bỏ sót vi phạm.

### Tracking

- ID switches
- Track continuity
- IDF1/HOTA nếu có ground-truth track ID

### Illegal Parking

- Violation Precision
- Violation Recall
- Violation F1
- False alarm
- Missed violation
- Sai số thời điểm phát hiện

---

## 19. Thứ tự triển khai

### Giai đoạn 1 - Detection

Train Faster R-CNN để khoanh đúng:

```text
car
motorcycle
```

**Kết quả:** `models/best.pth`

### Giai đoạn 2 - Video inference

Đưa `input/test.mp4` vào và xuất video có bounding box.

### Giai đoạn 3 - Tracking

Tích hợp ByteTrack và hiển thị Track ID.

### Giai đoạn 4 - ROI

Vẽ vùng cấm và xác định xe nằm trong vùng.

### Giai đoạn 5 - Motion + Timer

Phân biệt `MOVING / STOPPED / PARKED`.

### Giai đoạn 6 - Violation

Kết hợp:

```text
ROI + STOPPED + T_PARK
```

để kết luận đỗ sai.

### Giai đoạn 7 - Demo

Xuất video hoàn chỉnh:

```text
input/test.mp4
        ↓
Faster R-CNN
        ↓
ByteTrack
        ↓
ROI + Motion + Timer
        ↓
output/result.mp4
```

---

## 20. Cấu hình chốt cho phiên bản đầu tiên

```text
Detector       : Faster R-CNN
Backbone       : MobileNetV3-Large + FPN
Classes        : car, motorcycle
num_classes    : 3 (background + 2 classes)
Initialization : random, không COCO/ImageNet pretrained
Tracking       : ByteTrack
Spatial rule   : Polygon No-Parking ROI
Motion         : Track displacement + smoothing
Temporal rule  : stopped duration > T_PARK
Decision       : Rule-based
Output         : annotated video + violation images + logs
```

Mục tiêu của phiên bản đầu tiên là **pipeline chạy đúng end-to-end**. Sau đó mới tuning confidence threshold, motion threshold, `T_PARK`, tracker parameters và các hyperparameter training dựa trên validation/test set.

---

## 21. Kiến trúc và tham số triển khai hiện tại

Phần này mô tả chính xác cấu hình đang được dùng trong source code và Docker
của dự án. Đây là cấu hình chuẩn để tái lập quá trình train, đánh giá và chạy
pipeline CCTV.

### 21.1 Kiến trúc end-to-end

```text
Camera CCTV / Video
        ↓
Đọc và tách từng frame
        ↓
Faster R-CNN MobileNetV3-Large FPN
        ↓
Detection: car / motorcycle
        ↓
Bounding box + class + confidence
        ↓
ByteTrack
        ↓
Track ID + lịch sử vị trí
        ↓
Kiểm tra polygon No-Parking ROI
        ↓
Phân tích chuyển động theo bottom-center
        ↓
MOVING / STOPPED
        ↓
Bộ đếm thời gian dừng trong ROI
        ↓
PARKED / VIOLATION
        ↓
Video kết quả + ảnh bằng chứng + CSV log
```

Detector chỉ học hai lớp đối tượng. Hệ thống không học lớp
`illegal_parking`:

```text
0 = background
1 = car
2 = motorcycle
```

Quyết định cuối cùng được tạo bằng luật:

```text
VIOLATION = InsideNoParkingROI
            AND Stopped
            AND StoppedDuration >= ParkingThreshold
```

### 21.2 Faster R-CNN MobileNetV3-Large FPN

Model được tạo bằng Backbone/FPN của dự án rồi truyền vào Faster R-CNN:

```python
backbone = MobileNetV3WithFPN()
model = FasterRCNN(backbone, num_classes=3, ...)
```

Luồng xử lý bên trong detector:

```text
Ảnh RGB
  ↓
MobileNetV3-Large backbone
  ↓
Feature Pyramid Network
  ↓
Multi-scale feature maps
  ↓
Region Proposal Network
  ├── Objectness classification
  └── Anchor box regression
  ↓
Region proposals
  ↓
RoI Align
  ↓
Fast R-CNN detection head
  ├── background / car / motorcycle
  └── final bounding-box regression
```

MobileNetV3-Large được dùng để giảm chi phí tính toán so với backbone
ResNet50. FPN tạo đặc trưng ở nhiều scale, giúp detector xử lý xe lớn ở gần
camera và xe nhỏ ở xa camera. Mô hình Torchvision chuẩn có khoảng 19,4 triệu
tham số trước khi thay classification head.

Classification head COCO gốc được thay bằng:

```python
FastRCNNPredictor(in_features, num_classes=3)
```

### 21.3 Input và tiền xử lý

| Tham số | Giá trị hiện tại |
|---|---:|
| Input color | RGB |
| Tensor dtype | `float32` |
| Tensor range | `[0, 1]` |
| `min_size` | 800 px |
| `max_size` | 1333 px |
| Số object classes | 2 |
| `num_classes` gồm background | 3 |

Ảnh PIL được chuyển thành tensor bằng `torchvision.transforms.functional`.
Torchvision tự resize ảnh sao cho cạnh ngắn hướng tới `800` pixel và cạnh dài
không vượt quá `1333` pixel.

### 21.4 Dataset hiện tại

| Split | Tổng ảnh | Ảnh có motorcycle | Ảnh chỉ có car |
|---|---:|---:|---:|
| Train | 7.905 | 276 | 7.629 |
| Validation | 989 | 35 | 954 |
| Test | 989 | 35 | 954 |
| Tổng | 9.883 | 346 | 9.537 |

Số bounding box toàn bộ dataset:

```text
Car boxes        : 102.837
Motorcycle boxes : 460
Tổng annotations : 103.297
```

Annotation đầu vào sử dụng COCO:

```json
{
  "image_id": 1,
  "category_id": 1,
  "bbox": [100.0, 120.0, 300.0, 180.0],
  "area": 54000.0,
  "iscrowd": 0
}
```

Dataset loader chuyển bounding box từ COCO `[x, y, width, height]` thành
`[x1, y1, x2, y2]` trước khi đưa vào Faster R-CNN.

### 21.5 Cân bằng lớp motorcycle

Dataset lệch lớp mạnh, vì vậy train loader sử dụng
`WeightedRandomSampler`:

```text
Ảnh chỉ có car    → sampling weight = 1.0
Ảnh có motorcycle → sampling weight = 8.0
```

Tham số Docker tương ứng:

```env
MOTORCYCLE_WEIGHT=8.0
```

Weighted sampling chỉ tăng xác suất chọn ảnh motorcycle trong mỗi epoch; nó
không tạo thêm bản sao ảnh trên ổ đĩa.

### 21.6 Tham số training

| Nhóm | Tham số | Giá trị hiện tại |
|---|---|---:|
| Model | Architecture | Faster R-CNN |
| Backbone | Architecture | MobileNetV3-Large + FPN |
| Weights | Initialization | Random, train toàn bộ từ đầu |
| Classes | `num_classes` | 3 |
| Input | `min_size` | 800 |
| Input | `max_size` | 1333 |
| Train | Epochs | 20 |
| Train | Batch size | 2 |
| Train | DataLoader workers | 4 trong Docker |
| Train | AMP | Bật trên CUDA |
| Sampling | Motorcycle weight | 8.0 |
| Optimizer | Loại | SGD |
| Optimizer | Learning rate | 0.005 |
| Optimizer | Momentum | 0.9 |
| Optimizer | Weight decay | 0.0005 |
| Scheduler | Loại | StepLR |
| Scheduler | Step size | 5 epoch |
| Scheduler | Gamma | 0.1 |
| Reproducibility | Split seed | 42 |

Learning-rate schedule với cấu hình 20 epoch:

```text
Epoch 1–5   : 0.005
Epoch 6–10  : 0.0005
Epoch 11–15 : 0.00005
Epoch 16–20 : 0.000005
```

Tổng loss:

```text
L_total = L_RPN_objectness
        + L_RPN_box_regression
        + L_ROI_classification
        + L_ROI_box_regression
```

AMP được triển khai bằng:

```python
torch.autocast(device_type="cuda")
torch.amp.GradScaler("cuda")
```

AMP giúp giảm VRAM và tận dụng Tensor Core. Nếu GPU hết bộ nhớ, giảm
`BATCH_SIZE` từ `2` xuống `1` trước khi thay đổi kích thước input.

### 21.7 Checkpoint

Checkpoint có `mAP@0.50` tốt nhất được lưu tại:

```text
models/best.pth
```

Checkpoint có validation loss tốt nhất được lưu riêng tại:

```text
models/best_loss.pth
```

Sau mỗi epoch, tập validation được chạy inference thêm một lần để tính `AP`,
`mAP@0.50`, `precision`, `recall` và `F1` cho từng lớp `car`, `motorcycle` và
macro-average toàn bộ lớp. Các chỉ số này được in trực tiếp ra terminal, lưu
trong từng phần tử của `models/best.history.json`, đồng thời metric của epoch
tốt nhất được lưu trong checkpoint. Đây là mAP tại một ngưỡng IoU, không phải
COCO mAP@[0.50:0.95].

Nội dung checkpoint:

```python
{
    "model": model.state_dict(),
    "architecture": "custom_mobilenet_v3_large_fpn_v1",
    "initialization": "scratch",
    "epoch": epoch,
    "val_loss": val_loss,
    "metrics": metrics,
    "selection_metric": "mAP@IoU",
    "class_names": {1: "car", 2: "motorcycle"},
}
```

Lịch sử loss được lưu tại:

```text
models/best.history.json
```

### 21.8 Tham số inference

| Tham số | Giá trị hiện tại |
|---|---:|
| Confidence threshold | 0.50 |
| NMS threshold | 0.50 |
| Input min size | 800 px |
| Input max size | 1333 px |
| Device priority | CUDA → MPS → CPU |

Prediction của detector có dạng:

```python
{
    "boxes": Tensor[N, 4],
    "labels": Tensor[N],
    "scores": Tensor[N],
}
```

Pipeline chỉ giữ detection thỏa mãn:

```text
score >= confidence_threshold
AND label in {car, motorcycle}
```

### 21.9 ByteTrack

ByteTrack nhận detection của từng frame và gán Track ID tạm thời:

```text
Frame 100 → car → Track ID 8
Frame 101 → car → Track ID 8
Frame 102 → car → Track ID 8
```

| Tham số | Giá trị hiện tại |
|---|---:|
| Track activation threshold | 0.25 |
| Lost track buffer | 30 frame |
| Minimum matching threshold | 0.80 |
| Frame rate mặc định | 30 FPS |

Frame rate thực tế được đọc từ video. Nếu video không cung cấp FPS hợp lệ,
pipeline sử dụng `30 FPS`. Track ID là ID tạm của tracker, không phải biển số
hoặc danh tính phương tiện.

### 21.10 No-Parking ROI

Mỗi camera dùng một polygon riêng:

```json
{
  "points": [
    [120, 300],
    [900, 300],
    [1100, 700],
    [60, 700]
  ]
}
```

Điểm đại diện của phương tiện là bottom-center của bounding box:

```text
x = (x1 + x2) / 2
y = y2
```

Bottom-center gần với điểm tiếp xúc mặt đường hơn tâm bounding box, nên phù hợp
cho việc kiểm tra xe có nằm trong vùng cấm hay không.

### 21.11 Motion Analysis

| Tham số | Giá trị hiện tại |
|---|---:|
| Motion threshold | 8 px |
| Confirmation time | 2 giây |
| History window | 4 giây |

Với mỗi Track ID, pipeline lưu lịch sử bottom-center theo timestamp. Track được
xem là đứng yên khi:

```text
observed_time >= 2 giây
AND maximum_displacement <= 8 px
```

Displacement Euclid:

```text
d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

Cửa sổ nhiều frame giúp giảm false movement do bounding-box jitter. Không kết
luận xe đứng yên chỉ từ hai frame liên tiếp.

### 21.12 Parking State Machine

```text
MOVING
   ↓ đứng yên đủ thời gian xác nhận
STOPPED
   ↓ dừng trong ROI đủ parking_seconds
PARKED / VIOLATION
```

| Tham số | Giá trị hiện tại |
|---|---:|
| Parking threshold | 10 giây |
| Lost-track timeout | 3 giây |

Quy tắc cập nhật:

```text
Xe di chuyển
→ MOVING
→ reset stopped timer

Xe đứng yên ngoài ROI
→ STOPPED
→ không tính thời gian vi phạm

Xe đứng yên trong ROI
→ bắt đầu stopped timer

Stopped duration >= 10 giây
→ PARKED
→ violation = True
```

Khi xe di chuyển lại hoặc rời ROI, timer của sự kiện hiện tại được reset.

### 21.13 Output hệ thống

Video đã annotate:

```text
output/videos/result.mp4
```

Ảnh bằng chứng khi sự kiện vi phạm bắt đầu:

```text
output/violations/violation_ID8_20261007_101530_123456.jpg
```

CSV log:

```text
output/logs/result.csv
```

Các cột log:

```text
video_time_s
frame
track_id
class
confidence
zone
motion_px
stopped_duration_s
state
violation
```

### 21.14 Cấu hình Docker mặc định

```env
GPU_ID=0
PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
SEED=42
VAL_RATIO=0.10
TEST_RATIO=0.10
EPOCHS=20
BATCH_SIZE=2
WORKERS=4
LEARNING_RATE=0.005
MOTORCYCLE_WEIGHT=8.0
```

Lệnh train:

```bash
docker compose build
docker compose --profile tools run --rm gpu-check
docker compose run --rm train
```

### 21.15 Bảng cấu hình chốt

```text
Detector              : Faster R-CNN
Backbone              : MobileNetV3-Large
Feature network       : FPN
Classes               : car, motorcycle
num_classes           : 3
Input size            : min 800, max 1333
Optimizer             : SGD
Learning rate         : 0.005
Momentum              : 0.9
Weight decay          : 0.0005
Epochs                : 20
Batch size            : 2
Scheduler             : StepLR(step_size=5, gamma=0.1)
Mixed precision       : CUDA AMP
Motorcycle weight     : 8.0
Detection confidence  : 0.50
NMS threshold         : 0.50
Tracker               : ByteTrack
Track activation      : 0.25
Track buffer          : 30 frame
Matching threshold    : 0.80
Motion threshold      : 8 px
Motion confirmation   : 2 giây
Motion history        : 4 giây
Parking threshold     : 10 giây
Lost-track timeout    : 3 giây
Spatial rule          : polygon ROI
Decision rule         : ROI + STOPPED + TIMER
```
