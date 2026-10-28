# Illegal Parking Detection

## Module tự xây dựng: MobileNetV3-Large + FPN

Phần học tập đầu tiên của Faster R-CNN nằm trong:

```text
models/backbone.py   # lấy C3/C4/C5 từ MobileNetV3-Large
models/fpn.py        # lateral 1x1, top-down và output 3x3
test_modules.py      # kiểm tra shape C3/C4/C5 và P3/P4/P5
```

Chạy kiểm thử không cần pretrained weights:

```bash
python test_modules.py
```

Hoặc chạy bằng image Docker của dự án:

```bash
docker compose run --rm --no-deps train python test_modules.py
```

Kết quả cuối phải là:

```text
PASS: backbone and FPN shape tests
```

Trước khi train dài, chạy một forward/backward thật trên dataset:

```bash
python smoke_test_detector.py \
  --images data/train/images \
  --annotations data/annotations/train.json \
  --device cuda --amp
```

Backbone và FPN tự xây dựng hiện đã được nối vào Faster R-CNN trong
`models/detector.py`. Anchor Generator, RPN, Multi-Scale RoI Align và Detection
Head dùng module chuẩn của Torchvision. Mặc định toàn bộ mạng khởi tạo ngẫu
nhiên; không tải COCO hoặc ImageNet weights. Checkpoint cũ vẫn được nhận diện
và nạp bằng kiến trúc Torchvision cũ để giữ tương thích.

Hệ thống phát hiện `car`/`motorcycle` bằng Faster R-CNN MobileNetV3-Large FPN,
theo dõi bằng ByteTrack và kết luận đỗ sai từ ba điều kiện: xe nằm trong polygon
cấm đỗ, đứng yên đủ ổn định và vượt quá thời gian cho phép.

Tài liệu thiết kế đầy đủ nằm tại
[`README_FasterRCNN_Parking.md`](README_FasterRCNN_Parking.md).
Phần kiến trúc và toàn bộ tham số triển khai hiện tại nằm ở mục 21 của tài liệu
này.

## Cài đặt

Khuyến nghị Python 3.10–3.12 (PyTorch/torchvision có thể chưa hỗ trợ ngay các bản
Python mới hơn).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Kiểm tra phần rule không cần model/dataset:

```bash
python -m unittest discover -v
```

## Chuẩn bị dữ liệu

Dự án đọc annotation COCO. Category có tên `car`, `motorcycle` (hoặc
`motorbike`) được tự động ánh xạ thành label 1 và 2; ID category gốc không cần
liên tiếp.

```text
data/
├── train/images/
├── val/images/
└── annotations/
    ├── train.json
    └── val.json
```

Nên tách train/val theo camera hoặc video, không tách ngẫu nhiên các frame liền
nhau để tránh data leakage.

## Train

```bash
python -m src.train \
  --train-images data/train/images \
  --train-annotations data/annotations/train.json \
  --val-images data/val/images \
  --val-annotations data/annotations/val.json \
  --epochs 20 --batch-size 2
```

`models/best.pth` là checkpoint có `mAP@0.50` cao nhất;
`models/best_loss.pth` có validation loss thấp nhất. Lịch sử được cập nhật sau
mỗi epoch tại `models/best.history.json`, nên các epoch hoàn thành không mất nếu
train bị dừng. Code khóa `weights=None`; không có tùy chọn bật COCO hoặc ImageNet
pretrained weights, vì vậy mọi lần train mới đều bắt đầu hoàn toàn từ đầu.

Sau mỗi epoch, chương trình còn chạy inference trên toàn bộ tập validation và
in/lưu các chỉ số tổng hợp cùng chỉ số riêng cho `car`, `motorcycle`:

- `AP` và `mAP` tại IoU mặc định `0.50`;
- `precision`, `recall` và `F1`;
- số ground-truth và số prediction của từng lớp.

Ngưỡng mặc định để dựng đường precision-recall là `0.05`. Có thể đổi bằng:

```bash
python -m src.train ... \
  --metrics-score-threshold 0.05 \
  --metrics-iou-threshold 0.50
```

Các giá trị `precision`, `recall`, `F1` ở cấp tổng hợp là macro-average của hai
lớp. `mAP` ở đây là AP trung bình tại một ngưỡng IoU (`mAP@0.50`), không phải
COCO mAP@[0.50:0.95]. Việc tính metric mỗi epoch cần thêm một lượt inference
validation nên thời gian train sẽ tăng.

## Đánh giá detector

```bash
python -m src.evaluate \
  --images data/val/images \
  --annotations data/annotations/val.json \
  --checkpoint models/best.pth \
  --output output/evaluation.json
```

Lệnh trả về precision, recall, F1, AP@IoU, mAP, latency và FPS. Mặc định dùng
score threshold thấp `0.05` khi tính AP để không làm mất đường precision-recall.

Kiểm tra nhanh trên một ảnh:

```bash
python -m src.detect input/test.jpg --output output/detection.jpg
```

## Chạy video end-to-end

Sao chép và chỉnh polygon mẫu theo độ phân giải/camera thực tế:

```bash
cp config/no_parking_roi.example.json config/no_parking_roi.json
python -m src.pipeline input/test.mp4 \
  --roi-file config/no_parking_roi.json \
  --parking-seconds 10 \
  --motion-threshold 8
```

Hoặc truyền polygon trực tiếp:

```bash
python -m src.pipeline input/test.mp4 --roi "120,300;900,300;1100,700;60,700"
```

Kết quả gồm:

- `output/videos/result.mp4`: video có ROI, bbox, ID và trạng thái;
- `output/logs/result.csv`: log theo frame;
- `output/violations/`: một ảnh chứng cứ khi mỗi sự kiện vi phạm bắt đầu.

Thiết bị được chọn theo thứ tự CUDA → Apple MPS → CPU. Có thể ép bằng
`--device cpu`, `--device cuda` hoặc `--device mps`.

## Demo đỗ xe đúng/sai trên một ảnh

Chế độ này phân loại theo vị trí điểm giữa cạnh dưới của bounding box so với
polygon **được phép đỗ**. Nó phù hợp để trình diễn trên ảnh tĩnh, nhưng không
thể chứng minh xe đang đứng yên như pipeline video.

```bash
python -m src.detect_parking_image input/demo_parking_scene.png \
  --allowed-roi-file config/demo_allowed_parking_roi.json \
  --checkpoint models/best.pth \
  --confidence 0.50 \
  --output output/demo_parking_result.jpg
```

Kết quả gồm ảnh có polygon, khung xanh `PARKING LEGAL`, khung đỏ
`PARKING ILLEGAL`, loại xe, confidence và bảng đếm. Báo cáo chi tiết được lưu
cùng tên với đuôi `.json`.

## Hiệu chỉnh camera

`--motion-threshold` là bán kính jitter tính bằng pixel. Hãy đo trên xe thực sự
đứng yên ở cả gần và xa camera. `--motion-confirm-seconds` ngăn kết luận từ chỉ
hai frame. Bộ đếm `--parking-seconds` chỉ chạy khi track vừa đứng yên vừa ở trong
ROI; xe di chuyển hoặc ra khỏi ROI sẽ reset một sự kiện. Các tham số ByteTrack
được mở qua `--track-activation-threshold`, `--track-buffer` và
`--matching-threshold`.

Xem toàn bộ tùy chọn bằng:

```bash
python -m src.pipeline --help
python -m src.train --help
python -m src.evaluate --help
```

## Train bằng Docker + NVIDIA GPU

Dataset được sắp xếp trực tiếp theo bố cục dự án trong `data/`. Các ảnh split
được tạo bằng hard link nên xuất hiện đầy đủ trong train/val/test nhưng không
chiếm thêm dung lượng. Docker train bằng CUDA + AMP, ghi checkpoint mAP tốt nhất
vào `models/best.pth` và checkpoint loss tốt nhất vào `models/best_loss.pth`.

Chuẩn bị lại dữ liệu khi cần:

```bash
python3 scripts/bdd100k/download_dataset.py --output data/source

python3 scripts/split_coco.py \
  --annotations data/source/annotations/instances.json \
  --images-dir data/source/images \
  --output-dir data/annotations \
  --materialize hardlink \
  --seed 42
```

Hai công cụ BDD100K gốc nằm trong `scripts/bdd100k/`. Mã nguồn thư viện BDD100K
được giữ tại `third_party/bdd100k/` để toàn bộ dự án chỉ phụ thuộc một thư mục
gốc duy nhất.

Yêu cầu trên máy train: Docker Engine, Docker Compose plugin, NVIDIA driver và
NVIDIA Container Toolkit. Từ thư mục `Camera CCTV`:

```bash
cp .env.example .env
# Sửa GPU_ID trong .env nếu cần, sau đó:
docker compose build
docker compose --profile tools run --rm gpu-check
docker compose run --rm train
```

Checkpoint và lịch sử train được ghi ra máy host:

```text
models/
├── detector.py
├── best.pth
├── best_loss.pth
└── best.history.json

data/annotations/
├── train.json
├── val.json
└── test.json
```

Có thể đổi `EPOCHS`, `BATCH_SIZE`, `WORKERS`, `LEARNING_RATE`,
`MOTORCYCLE_WEIGHT` và `GPU_ID` trong `.env`. Nếu GPU hết VRAM, giảm
`BATCH_SIZE` xuống `1`. Dataset hiện lệch lớp mạnh nên Docker mặc định đặt
`MOTORCYCLE_WEIGHT=8.0`; đây là weighted sampling, không sao chép ảnh.
Nếu driver/GPU cần bản CUDA khác, đổi `PYTORCH_IMAGE` trong `.env` sang tag
PyTorch CUDA tương thích rồi build lại.
