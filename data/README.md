# Data

Tài liệu liên quan: [Research §8 — yêu cầu cho dữ liệu](../docs/research.md) · [Architecture](../docs/architecture.md).

## 1. Policy

- Chỉ dùng dữ liệu **synthetic** sinh bởi code trong `ml/data/`.
- **Không** dùng dữ liệu nội bộ hay dữ liệu thật của bất kỳ công ty nào. Nếu sau này bổ sung public dataset,
  phải ghi rõ nguồn, license và cách xử lý tại đây.
- **Không sinh PII**: không tên, số điện thoại, biển số hay thông tin định danh cá nhân.
- Dữ liệu **không được commit** vào git (`data/raw/*`, `data/processed/*`, `data/oracle/` đã được ignore).
  Mọi file phải tái tạo được từ code + config + random seed.

## 2. Cách generate

Chạy từ thư mục gốc của repository (sau khi `pip install -r requirements-dev.txt`):

```bash
python -m ml.data.generate                                        # kích thước mặc định, seed 42
python -m ml.data.generate --n-bookings 20000 --n-drivers 5000 --seed 7
```

Các tham số khác nằm trong [`ml/data/config.py`](../ml/data/config.py). Trên máy dev (i7-12700H, 16 GB RAM), cấu hình mặc định
chạy **1.8 giây**, peak working set **379 MiB**.

## 3. Layout

| Thư mục | File | Ai được đọc |
|---|---|---|
| `raw/` | `drivers.parquet`, `bookings.parquet`, `candidates.parquet`, `metadata.json` | Mọi code: EDA, feature engineering, training, seed database |
| `oracle/` | `candidate_truth.parquet`, `driver_latents.parquet`, `metadata.json` | **Chỉ offline evaluation** (Phase 4, 6, 20). Không bao giờ dùng cho training, feature hay serving |
| `processed/` | Feature tables, train/validation/test splits | Phase 5 |

## 4. Schema

### 4.1 `raw/drivers.parquet` — 20,000 dòng

| Column | Type | Mô tả |
|---|---|---|
| `driver_id` | int32 | Primary key |
| `vehicle_type` | category | `car_4` hoặc `car_7` |
| `rating` | float64 | Rating trung bình; **NaN** khi `completed_trips < 5` (chưa đủ đánh giá) |
| `acceptance_rate` | float64 | Tỷ lệ nhận offer lịch sử, tính **trước** cửa sổ mô phỏng |
| `cancellation_rate` | float64 | Tỷ lệ huỷ lịch sử, tính trước cửa sổ mô phỏng |
| `completed_trips` | int32 | Số chuyến đã hoàn thành |
| `idle_time_min` | float64 | Snapshot hiện tại: số phút rảnh (dùng cho demo live matching) |
| `current_lat`, `current_lon` | float64 | Snapshot vị trí hiện tại |

### 4.2 `raw/bookings.parquet` — 100,000 dòng

| Column | Type | Mô tả |
|---|---|---|
| `booking_id` | int32 | Primary key, tăng dần theo `request_time` |
| `request_time` | datetime64[us] | Giờ địa phương giả lập, 2026-06-01 → 2026-07-26 |
| `pickup_lat`, `pickup_lon` | float64 | Điểm đón |
| `destination_lat`, `destination_lon` | float64 | Điểm trả (cách điểm đón ≥ 1 km) |
| `passenger_type` | category | `individual`, `group` (group cần xe `car_7`) |
| `traffic_level` | category | `low`, `medium`, `high` |
| `weather` | category | `clear`, `rain`, `heavy_rain` |
| `time_of_day` | category | `night` (0–5h), `morning_peak` (6–9h), `midday` (10–15h), `evening_peak` (16–19h), `evening` (20–23h) |

### 4.3 `raw/candidates.parquet` — 780,640 dòng (matching observations)

Mỗi dòng là một cặp (booking, tài xế available, đúng loại xe, trong bán kính 5 km). Khoá: (`booking_id`, `driver_id`).

| Column | Type | Mô tả |
|---|---|---|
| `booking_id`, `driver_id` | int32 | Foreign keys |
| `driver_lat`, `driver_lon` | float64 | Vị trí tài xế tại thời điểm booking |
| `distance_km` | float64 | Haversine tới điểm đón (≤ 5 km) |
| `estimated_eta_min` | float64 | ETA đón khách ước tính |
| `idle_time_min` | float64 | Số phút rảnh tại thời điểm booking |
| `logging_policy` | category | Policy đã sinh log cho booking này: `nearest` hoặc `explore` |
| `offer_rank` | Int8 (nullable) | Thứ tự offer 1..3 nếu tài xế được offer; `<NA>` nếu không |
| `accepted` | boolean (nullable) | Nhãn — chỉ có khi được offer |
| `cancelled` | boolean (nullable) | Chỉ có khi `accepted = True` |

Đối chiếu với field trong yêu cầu project: `distance` → `distance_km`, `estimated_eta` → `estimated_eta_min`,
`driver_features` → join `drivers` theo `driver_id` + `idle_time_min`.

### 4.4 `oracle/` — sự thật ẩn

| File | Columns | Mô tả |
|---|---|---|
| `candidate_truth.parquet` | `booking_id`, `driver_id`, `p_accept`, `u_accept`, `p_cancel`, `u_cancel` | Cùng thứ tự dòng với `candidates.parquet`. Tài xế nhận ⇔ `u_accept < p_accept`; chuyến bị huỷ ⇔ `u_cancel < p_cancel` |
| `driver_latents.parquet` | `driver_id`, `selectivity`, `quality`, `unreliability` | Đặc điểm ẩn của tài xế |

---

## 5. Mô hình sinh dữ liệu

### 5.1 Thành phố, thời gian, ngữ cảnh

- **Thành phố hư cấu** 20 × 20 km quanh toạ độ (0, 0) — cố ý không gắn với thành phố thật. 70% điểm đón/trả/tài xế rơi vào 4 hotspot, 30% phân bố đều:

  | Hotspot | Tâm (x, y) km | σ km | Trọng số |
  |---|---|---|---|
  | Downtown | (0, 0) | 2.0 | 0.30 |
  | Business district | (5, 4) | 1.5 | 0.15 |
  | Residential area | (−6, −3) | 2.0 | 0.15 |
  | Transport hub | (7, −7) | 0.8 | 0.10 |

- **Thời gian:** 56 ngày từ 2026-06-01. Nhu cầu theo giờ có đỉnh 7–9h và 16–19h; cuối tuần × 0.85.
- **Thời tiết:** chuỗi Markov 3 trạng thái theo giờ, chung cho cả thành phố.

  | Từ \ Sang | clear | rain | heavy_rain |
  |---|---|---|---|
  | clear | 0.93 | 0.06 | 0.01 |
  | rain | 0.25 | 0.65 | 0.10 |
  | heavy_rain | 0.15 | 0.45 | 0.40 |

- **Traffic:** `score = base(time_of_day) + weather + 0.6 · centrality + N(0, 0.7)`, với base: night −1.5, morning_peak 1.0, midday 0,
  evening_peak 1.2, evening −0.3; weather: rain +0.5, heavy_rain +1.0. Ngưỡng: score < −0.3 → low; < 1.3 → medium; còn lại → high.
- **Hành khách:** 8% booking là `group`.

### 5.2 Tài xế

- 25% `car_7`. 15% là tài xế mới (0–49 chuyến); còn lại `completed_trips ~ lognormal(6.0, 1.1)`.
- **Đặc điểm ẩn:** `selectivity ~ N(0,1)`, `quality ~ N(0,1)`, `unreliability = −0.4 · quality + √0.84 · N(0,1)`.
- **Thống kê lịch sử là ước lượng có nhiễu của đặc điểm ẩn** (tài xế ít chuyến → nhiễu hơn, giống thống kê thật trên mẫu nhỏ):
  - `acceptance_rate = Binomial(n, σ(0.6 − selectivity)) / n`, với `n = max(10, 1.6 · completed_trips)`
  - `cancellation_rate = Binomial(n, σ(−2.6 + 0.8 · unreliability)) / n`, với `n = max(5, completed_trips)`
  - `rating`: `4.65 + 0.2 · quality` được co về 4.7 với trọng số 20 chuyến, cộng nhiễu N(0, 0.03); NaN nếu < 5 chuyến.

### 5.3 Ứng viên (snapshot theo từng booking)

- Số tài xế available gần điểm đón `~ Poisson(12 × supply(time_of_day) × supply(weather) × (0.6 + 0.6 · e^{−(r/7)²}))`,
  với `r` là khoảng cách điểm đón tới trung tâm. Supply: night 0.6, morning_peak 0.7, midday 1.1, evening_peak 0.65, evening 0.9;
  clear 1.0, rain 0.8, heavy_rain 0.55. Booking `group` chỉ giữ `Binomial(n, 0.25)` tài xế `car_7`. Tối đa 30 ứng viên.
- Tài xế được lấy ngẫu nhiên từ pool phù hợp, bỏ trùng trong cùng booking; vị trí phân bố đều trong đĩa bán kính 5 km.
- `estimated_eta_min = 1 + distance_km × detour / speed × 60`, với `detour = 1.3 · e^{N(0, 0.1)}` và speed low 32 / medium 22 / high 14 km/h.
- `idle_time_min ~ Exponential(mean)` với mean: night 20, morning_peak 6, midday 12, evening_peak 6, evening 10 (tối đa 180).

### 5.4 Mô hình hành vi ẩn ([`ml/data/behavior.py`](../ml/data/behavior.py))

**Nhận chuyến:** `p_accept = σ(z)`, với `z` là tổng các thành phần:

| Thành phần | Hệ số | Ý nghĩa |
|---|---|---|
| Intercept | +2.1 | Hiệu chỉnh theo tỷ lệ tổng quát (§5.7) |
| ETA (phút) | −0.18 | Đón càng lâu càng ít muốn nhận |
| max(ETA − 8, 0) | −0.10 | Phạt thêm khi đón quá 8 phút (phi tuyến) |
| log(1 + trip_km) | +0.45 | Chuyến dài hấp dẫn hơn |
| min(distance_km / (trip_km + 0.5), 3) | −0.90 | Quãng đón dài cho chuyến ngắn |
| selectivity (ẩn) | −1.00 | Tài xế kén chọn |
| 1 − e^{−idle/15} | +0.90 | Rảnh lâu → dễ nhận, có bão hoà |
| rain / heavy_rain | −0.25 / −0.60 | |
| heavy_rain × ETA | −0.06 | Tương tác |
| traffic medium / high | −0.10 / −0.35 | |
| high traffic × log(1 + trip_km) | −0.15 | Tương tác |
| night × ETA | −0.05 | Tương tác |
| completed_trips < 100 | +0.35 | Tài xế mới dễ nhận |
| group booking | +0.25 | |
| quality (ẩn) | −0.10 | Tài xế tốt kén hơn một chút |
| Nhiễu theo từng offer | N(0, 0.6) | Yếu tố không quan sát được |

**Huỷ chuyến (sau khi đã nhận):** `p_cancel = σ(−3.0 + 0.09 · ETA + 0.8 · unreliability + 0.15 · rain + 0.35 · heavy_rain + 0.20 · night + N(0, 0.4))`.
ETA dài làm tăng huỷ → tạo trade-off thật giữa việc chọn tài xế xa hơn và Cancellation Rate.

**Vì sao có dạng này:** có phi tuyến và tương tác (để so sánh Logistic Regression với XGBoost có ý nghĩa); đặc điểm ẩn chỉ quan sát được gián tiếp
qua thống kê lịch sử; có nhiễu theo từng offer → dữ liệu không thể tách biệt hoàn hảo.

### 5.5 Logging policy — dữ liệu training

Log lịch sử được sinh bằng một policy **khác** với các policy sẽ được đánh giá:

1. Mỗi booking: 80% offer theo thứ tự gần nhất trước (`nearest`), 20% theo thứ tự ngẫu nhiên (`explore`).
2. Offer lần lượt, dừng khi có người nhận hoặc đã gửi 3 offer.
3. Chỉ dòng được offer mới có nhãn — giống log thật. Exploration đảm bảo có nhãn cho cả tài xế không phải gần nhất.

**Selection bias quan sát được ngay trong dữ liệu:** xác suất nhận thật trung bình của tài xế **được offer** là 0.540, cao hơn so với toàn bộ ứng viên (0.431),
vì log ưu tiên tài xế gần. Model train trên log sẽ thấy nhiều tài xế gần hơn tỷ lệ thực tế.

### 5.6 Common random numbers & oracle

`u_accept`, `u_cancel` được sinh một lần cho mỗi cặp (booking, driver). Mọi policy được đánh giá ở Phase 4, 6 (Nearest Driver, ML, Oracle...)
dùng chung các số này → cùng một "thế giới", chênh lệch metric chỉ đến từ thứ tự offer ([research §7.2](../docs/research.md)).

### 5.7 Hiệu chỉnh tham số

Hệ số được chọn **trước** theo chiều và độ lớn hợp lý. Chỉ hai tham số được hiệu chỉnh, dựa trên **khoảng hợp lý của tỷ lệ tổng quát được đặt trước khi
có bất kỳ model nào**. Trong quá trình hiệu chỉnh không tính metric nào của baseline hay ML.

| Tham số | Lý do chỉnh | Trước | Sau |
|---|---|---|---|
| Ngưỡng traffic high | Traffic `high` 51.0% ngoài khoảng 25–40% | 0.9 | 1.3 → 36.4% |
| Acceptance intercept | Offer acceptance của log ngoài khoảng 0.45–0.65 | 1.4 | 2.1 → 0.542 |

Quét intercept (sau khi đã sửa ngưỡng traffic), chọn giá trị gần giữa khoảng (0.55) nhất:

| Intercept | 1.4 | 1.7 | 1.9 | **2.1** | 2.3 |
|---|---|---|---|---|---|
| Offer acceptance (log) | 0.441 | 0.484 | 0.513 | **0.542** | 0.571 |
| Cancellation rate (log) | 0.129 | 0.130 | 0.131 | **0.131** | 0.132 |
| Oracle AUC | 0.893 | 0.897 | 0.899 | **0.902** | 0.904 |

---

## 6. Thống kê dataset mặc định (seed 42)

Số liệu từ `raw/metadata.json`, `oracle/metadata.json` và script kiểm tra ở Phase 2.

### 6.1 Kích thước

| | Giá trị |
|---|---|
| Drivers / bookings / candidates | 20,000 / 100,000 / 780,640 |
| Logged offers (rank 1 / 2 / 3) | 149,944 (98,564 / 33,037 / 18,343) |
| Ứng viên mỗi booking: mean / p50 / p95 / max | 7.81 / 7 / 16 / 30 |
| Booking không có ứng viên | 1.44% |
| Driver thiếu rating | 1.46% (292) |
| Booking dùng exploration | 20.2% |
| File size: drivers / bookings / candidates / candidate_truth / driver_latents | 0.61 / 4.45 / 17.60 / 27.43 / 0.68 MiB |

### 6.2 Ngữ cảnh (% booking)

| | |
|---|---|
| time_of_day | night 5.6 · morning_peak 24.1 · midday 28.3 · evening_peak 29.0 · evening 13.1 |
| weather | clear 74.0 · rain 22.3 · heavy_rain 3.8 |
| traffic_level | low 14.1 · medium 49.5 · high 36.4 |
| passenger_type | individual 92.0 · group 8.0 |
| vehicle_type (% driver) | car_4 74.9 · car_7 25.1 |

### 6.3 Phân phối

| Biến | p5 | p25 | p50 | p75 | p95 | mean |
|---|---|---|---|---|---|---|
| drivers.rating (không NaN) | 4.35 | 4.54 | 4.66 | 4.77 | 4.94 | 4.65 |
| drivers.acceptance_rate | 0.26 | 0.48 | 0.64 | 0.78 | 0.91 | 0.62 |
| drivers.cancellation_rate | 0.00 | 0.04 | 0.07 | 0.12 | 0.23 | 0.09 |
| drivers.completed_trips | 16 | 112 | 320 | 742 | 2266 | 640 |
| bookings trip distance (km) | 2.11 | 5.20 | 8.44 | 11.69 | 16.24 | 8.67 |
| candidates.distance_km | 1.12 | 2.50 | 3.54 | 4.33 | 4.87 | 3.34 |
| candidates.estimated_eta_min | 5.00 | 9.92 | 13.93 | 18.21 | 26.76 | 14.53 |
| candidates.idle_time_min | 0.40 | 2.40 | 6.00 | 12.50 | 30.10 | 9.40 |

### 6.4 Hiệu quả của logging policy

Đây là hiệu quả của **policy đã sinh log** (80% nearest + 20% explore). **Không phải** kết quả của baseline ở Phase 4 hay của ML.

| Metric | Giá trị |
|---|---|
| Offer acceptance rate | 0.542 |
| Matching success rate (≤ 3 offer) | 0.812 |
| Average ETA của tài xế đã nhận | 7.99 phút |
| Cancellation rate | 0.131 |
| Số offer trung bình / booking có ứng viên | 1.52 |

### 6.5 Oracle (chỉ để kiểm tra độ khó của dữ liệu)

| Metric | Giá trị |
|---|---|
| Mean p_accept: mọi ứng viên / tài xế được offer | 0.431 / 0.540 |
| Oracle ROC-AUC (accept, trên logged offers) | 0.902 |
| Oracle ROC-AUC (cancel, trên chuyến đã nhận) | 0.745 |

Oracle AUC là **cận trên** cho mọi model vì oracle biết cả đặc điểm ẩn lẫn nhiễu từng offer. Model thật chỉ thấy feature quan sát được nên sẽ thấp hơn.

---

## 7. Reproducibility

- Cùng `generator_version` + config → **file giống hệt từng byte**. Đã kiểm chứng: 2 lần chạy cấu hình mặc định cho SHA-256 trùng khớp ở cả 7 file.
- Mỗi giai đoạn (drivers, weather, bookings, candidates, outcomes, logging) dùng một luồng random riêng từ `SeedSequence.spawn`,
  nên đổi số booking không làm thay đổi dữ liệu tài xế.
- `raw/metadata.json` lưu config; `oracle/metadata.json` lưu tham số hành vi ẩn.

## 8. Quy tắc chống leakage cho các phase sau

| Không được | Lý do |
|---|---|
| Đọc `data/oracle/` trong feature engineering, training hoặc serving | Oracle leakage |
| Dùng `cancelled` làm feature để dự đoán `accepted` | Chỉ có sau khi đã nhận |
| Dùng `offer_rank`, `logging_policy` làm feature | Là sản phẩm của log, không tồn tại khi xếp hạng một booking mới |
| Chia train/test ngẫu nhiên theo dòng | Candidate của cùng booking lọt sang cả hai tập; phải chia theo thời gian ở cấp booking |

## 9. Giả định & giới hạn

- **Snapshot theo booking (PRD A4):** không theo dõi tài xế theo thời gian; một tài xế có thể xuất hiện ở các booking trùng giờ; không mô phỏng supply cạn dần.
- Thống kê lịch sử của tài xế cố định trong cả cửa sổ 56 ngày (không có drift).
- Không có mạng lưới đường: distance đường chim bay + detour ngẫu nhiên.
- Thời tiết chung toàn thành phố; traffic ở cấp booking.
- Mô hình hành vi là **giả định do project tự đặt ra** — mọi kết luận chỉ đúng trong phạm vi simulator này.
