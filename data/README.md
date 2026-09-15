# Data

## Policy

- Chỉ dùng dữ liệu **synthetic** sinh bởi code trong `ml/data/`.
- **Không** dùng dữ liệu nội bộ hay dữ liệu thật của bất kỳ công ty nào. Nếu sau này bổ sung public dataset,
  phải ghi rõ nguồn, license và cách xử lý tại đây.
- **Không sinh PII**: không tên, số điện thoại, biển số hay thông tin định danh cá nhân.
- Dữ liệu **không được commit** vào git (`data/raw/*`, `data/processed/*` đã được ignore).
  Mọi file phải tái tạo được từ code + config + random seed.

## Layout

| Thư mục | Nội dung | Phase |
|---|---|---|
| `raw/` | Output trực tiếp của generator: drivers, bookings, matching observations | 2 |
| `processed/` | Feature tables và train/validation/test splits | 5 |

## Cách generate

Được document ở Phase 2: lệnh chạy, config, seed, schema từng file, cách sinh ground truth và các giả định của simulator.
