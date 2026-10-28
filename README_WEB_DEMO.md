# Web demo phát hiện đỗ xe sai quy định

Web chạy trực tiếp trên server nội bộ và không gửi video/checkpoint ra Internet.
Backend gọi `src.pipeline` để thực hiện detection, ByteTrack, phân tích chuyển
động và luật thời gian đỗ.

## Chức năng

- sử dụng video mẫu hoặc tải video riêng lên;
- vẽ polygon vùng cấm đỗ trực tiếp trên video;
- cấu hình confidence, thời gian đỗ và ngưỡng chuyển động;
- theo dõi tiến độ và log xử lý;
- xem video kết quả ngay trong trình duyệt;
- hiển thị tổng số track và xe vi phạm;
- tải video, CSV và xem ảnh bằng chứng.

## Chạy an toàn bằng CPU

```bash
cd ~/illegal-parking-detection
WEB_DEVICE=cpu WEB_PORT=8501 docker compose up web
```

Truy cập từ Mac cùng mạng:

```text
http://192.168.1.111:8501
```

Dừng web bằng `Ctrl+C`, hoặc chạy từ terminal khác:

```bash
cd ~/illegal-parking-detection
docker compose stop web
```

## Chạy bằng GPU

Trước tiên kiểm tra GPU:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
```

Ví dụ chạy trên GPU vật lý số 1:

```bash
cd ~/illegal-parking-detection

GPU_ID=1 docker compose run --rm --no-deps \
  -p 8501:8501 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$PWD:/workspace" \
  train \
  python web_demo.py --host 0.0.0.0 --port 8501 --device cuda
```

Trong container, GPU vật lý số 1 được ánh xạ thành `cuda:0`.

## Dữ liệu đầu ra

Mỗi lần chạy tạo một thư mục riêng:

```text
output/web/<job_id>/
├── videos/
│   ├── result_raw.mp4
│   └── result.mp4
├── logs/
│   └── result_raw.csv
└── violations/
    └── violation_ID*.jpg
```

`result.mp4` được chuyển sang H.264 để Safari/Chrome có thể phát trực tiếp.

## Lưu ý

- Web chỉ cho phép một job inference chạy tại một thời điểm.
- Giới hạn mỗi request tải lên là 512 MB.
- Chỉ nên mở cổng `8501` trong mạng nội bộ.
- Không dừng tiến trình GPU của người khác nếu GPU đang đầy; hãy dùng CPU hoặc
  chọn GPU khác.
