# SmartMatch AI — Evaluation

| | |
|---|---|
| Version | 0.2 |
| Phase | 4 — Baseline; §6 cập nhật ở Phase 5 (Phase 6 bổ sung ML vs Baseline) |
| Status | Baseline done — chờ review |
| Last updated | 2026-09-16 |

Tài liệu liên quan: [PRD §8 — Metrics](product-requirements.md) · [Research §6–7](research.md) · [Data](../data/README.md) ·
kết quả gốc: [`ml/evaluation/results/baseline.json`](../ml/evaluation/results/baseline.json).

> Mọi con số trong tài liệu này được sinh bởi `python -m ml.evaluation.run_baseline` trên dataset mặc định (generator 1.0.0, seed 42).
> Kết quả chỉ đúng trong giả định của simulator (§7).

---

## 1. Tóm tắt Phase 4

Trên **tập test** (13,946 booking, 2026-07-19 → 2026-07-26):

| | Nearest Driver (baseline) |
|---|---|
| Matching success rate | **85.6%** |
| Acceptance rate (offer-level) | **62.1%** |
| Average ETA của tài xế đã nhận | **7.28 phút** |
| Cancellation rate | **11.9%** |

Ba phát hiện quan trọng cho Phase 5–6:

1. **Trần cải thiện của Matching Success Rate rất thấp.** Ngay cả Oracle (biết xác suất nhận thật) chỉ hơn Nearest Driver **+1.2 pp**
   (95% CI [+1.0, +1.5]). Dư địa lớn hơn nằm ở hiệu suất dispatch: first-offer acceptance +7.4 pp, số offer mỗi booking −0.10.
2. **Xếp hạng chỉ theo xác suất nhận — kể cả hoàn hảo — vi phạm guardrail.** Oracle chọn tài xế xa hơn: ETA **+1.29 phút (≈ +17.6%, vượt
   ngưỡng +15% của H2)**, cancellation **+1.3 pp** (vi phạm H3), completed rate **không cải thiện** (CI [−0.6, +0.4] pp).
3. **Một rule đặt tay đã thắng Nearest Driver** ở Matching Success Rate (+0.8 pp), completed rate (+1.3 pp) và cancellation (−0.7 pp),
   với ETA chỉ tăng ≈ +6.3%. → ML phải được so với cả weighted rule, không chỉ Nearest Driver.

---

## 2. Protocol

### 2.1 Dữ liệu & split

- Split theo thời gian ở cấp booking (ADR-011): **validation** 2026-07-11 → 07-18 (14,191 booking),
  **test** 2026-07-19 → 07-26 (13,946 booking). Các baseline không cần train nên tập train không được dùng ở Phase 4.
- Candidate set: snapshot ứng viên của Phase 2 (tài xế available, đúng loại xe, trong bán kính 5 km). **Mọi policy dùng đúng tập này** —
  chỉ thứ tự offer khác nhau. Việc lọc tài xế trực tiếp từ database thuộc Phase 7–9.

### 2.2 Replay offer tuần tự

1. Mỗi policy cho một score cho từng ứng viên; offer theo score giảm dần, hoà thì theo `driver_id`.
2. Offer lần lượt tối đa **N = 3** lượt, dừng khi có người nhận.
3. Tài xế nhận ⇔ `u_accept < p_accept`; chuyến bị huỷ ⇔ đã nhận và `u_cancel < p_cancel`. Các số ngẫu nhiên `u` cố định cho từng cặp
   (booking, driver) — **common random numbers** — nên mọi policy được đánh giá trong cùng một "thế giới".
4. Policy chỉ nhận các cột có tại thời điểm quyết định (candidate + booking context + driver profile); không thấy nhãn, `offer_rank`,
   `logging_policy` hay dữ liệu oracle. Riêng Oracle đọc `p_accept` và chỉ tồn tại trong module evaluation.

Kiểm chứng tự động: replay Nearest Driver **tái tạo chính xác** từng offer, lượt nhận và lượt huỷ trong log của các booking dùng policy
`nearest` (`ml/tests/test_offline_eval.py`).

### 2.3 Định nghĩa metric (chốt ở Phase 4, dùng giống hệt cho Baseline và ML)

| Metric | Công thức | Ghi chú |
|---|---|---|
| **Matching success rate (MSR)** | booking có người nhận trong ≤ N offer / **mọi** booking của split | Booking không có ứng viên tính là thất bại. North Star (PRD §8.2) |
| **Acceptance rate (AR)** | offer được nhận / offer đã gửi | Mỗi booking matched có đúng một offer được nhận |
| **Average ETA (matched)** | trung bình ETA của tài xế đã nhận, trên các booking matched | **Selection bias**: luôn đọc cùng MSR |
| P90 ETA (matched) | phân vị 90 của ETA trên booking matched | Đuôi phân phối trải nghiệm chờ |
| **Cancellation rate (CR)** | booking bị huỷ / booking matched | Guardrail (H3) |
| **Completed rate** | booking matched và không bị huỷ / mọi booking | Kết hợp MSR và CR. North Star thứ hai từ Phase 5 (PRD §8.2) |
| First-offer acceptance rate | booking nhận ngay ở offer đầu / booking có ứng viên | Tốc độ xác nhận |
| Offers per booking | tổng offer / booking có ứng viên | Hiệu suất dispatch |

### 2.4 Độ tin cậy

**Paired percentile bootstrap**: lấy mẫu lại có hoàn lại toàn bộ booking của split 1,000 lần (seed 2026); mọi policy được đo trên **cùng**
mẫu, rồi lấy phân vị 2.5–97.5 của chênh lệch so với Nearest Driver. Một khác biệt được coi là có ý nghĩa khi khoảng tin cậy **không chứa 0**.

### 2.5 Các policy

| Policy | Vai trò | Định nghĩa |
|---|---|---|
| Random | Mức sàn — sanity check | Thứ tự ngẫu nhiên trong candidate set (seed 7) |
| **Nearest Driver** | **Baseline chính, mốc so sánh** | Khoảng cách đường chim bay tới điểm đón tăng dần |
| Weighted rule | Baseline phụ | Điểm theo trọng số đặt tay (bảng dưới), **cố định trước khi chạy đánh giá, không tinh chỉnh theo kết quả** |
| Oracle | Mức trần, không đạt được ngoài đời | Xác suất nhận thật `p_accept` của simulator giảm dần |

Trọng số weighted rule (đơn vị "điểm"; 1 phút ETA = −0.2 điểm):

| Thành phần | Trọng số | Tương đương 1 phút ETA |
|---|---|---|
| `estimated_eta_min` | −0.2 / phút | — |
| `acceptance_rate` | +2.0 | +10 pp acceptance lịch sử |
| `cancellation_rate` | −2.0 | −10 pp cancellation lịch sử |
| `rating` − 4.7 (thiếu rating tính là 4.7) | +0.5 / sao | +0.4 sao |
| `idle_time_min` (chặn ở 30 phút) | +0.02 / phút | +10 phút rảnh |

---

## 3. Kết quả baseline — tập test

13,946 booking (13,788 có ứng viên), 111,046 dòng ứng viên. Trong ngoặc: chênh lệch so với Nearest Driver, 95% CI.

| Metric | Random | **Nearest Driver** | Weighted rule | Oracle |
|---|---|---|---|---|
| Matching success rate | 73.5% ([−12.8, −11.6] pp) | **85.6%** | 86.4% ([+0.6, +1.0] pp) | 86.8% ([+1.0, +1.5] pp) |
| Acceptance rate | 39.4% ([−23.4, −21.9] pp) | **62.1%** | 65.4% ([+2.9, +3.7] pp) | 68.0% ([+5.4, +6.3] pp) |
| Average ETA, matched (phút) | 10.75 ([+3.37, +3.55]) | **7.28** | 7.75 ([+0.43, +0.49]) | 8.57 ([+1.24, +1.33]) |
| P90 ETA, matched (phút) | 16.83 | **12.65** | 13.19 | 14.08 |
| Cancellation rate | 16.1% ([+3.5, +5.0] pp) | **11.9%** | 11.2% ([−1.1, −0.3] pp) | 13.2% ([+0.9, +1.9] pp) |
| Completed rate | 61.6% ([−14.6, −13.1] pp) | **75.5%** | 76.8% ([+0.9, +1.7] pp) | 75.3% ([−0.6, +0.4] pp) |
| First-offer acceptance rate | 42.7% ([−30.4, −28.6] pp) | **72.2%** | 76.5% ([+3.8, +4.8] pp) | 79.6% ([+6.8, +8.0] pp) |
| Offers per booking | 1.885 ([+0.476, +0.505]) | **1.395** | 1.336 ([−0.065, −0.052]) | 1.293 ([−0.110, −0.094]) |

## 4. Kết quả baseline — tập validation

14,191 booking (13,977 có ứng viên), 111,260 dòng ứng viên. Dùng cho phát triển model ở Phase 5 mà không chạm vào test.

| Metric | Random | **Nearest Driver** | Weighted rule | Oracle |
|---|---|---|---|---|
| Matching success rate | 71.1% ([−13.3, −12.1] pp) | **83.8%** | 84.6% ([+0.6, +1.1] pp) | 85.2% ([+1.2, +1.7] pp) |
| Acceptance rate | 37.8% ([−22.6, −21.2] pp) | **59.6%** | 63.0% ([+3.0, +3.8] pp) | 65.7% ([+5.5, +6.5] pp) |
| Average ETA, matched (phút) | 10.90 ([+3.43, +3.61]) | **7.38** | 7.83 ([+0.42, +0.48]) | 8.62 ([+1.20, +1.29]) |
| P90 ETA, matched (phút) | 17.03 | **12.78** | 13.36 | 14.30 |
| Cancellation rate | 16.6% ([+3.6, +5.1] pp) | **12.3%** | 11.8% ([−0.8, −0.0] pp) | 13.8% ([+0.9, +2.0] pp) |
| Completed rate | 59.3% ([−15.0, −13.5] pp) | **73.5%** | 74.6% ([+0.7, +1.5] pp) | 73.5% ([−0.5, +0.5] pp) |
| First-offer acceptance rate | 41.7% ([−29.4, −27.6] pp) | **70.1%** | 74.6% ([+4.0, +5.0] pp) | 77.8% ([+7.0, +8.3] pp) |
| Offers per booking | 1.911 ([+0.471, +0.499]) | **1.426** | 1.364 ([−0.069, −0.056]) | 1.318 ([−0.117, −0.100]) |

---

## 5. Diễn giải

1. **Sanity check đạt.** Random kém Nearest Driver ở mọi metric (MSR −12.1 pp trên test) — khoảng cách thực sự quan trọng, đúng với EDA.
2. **MSR gần như đã bão hoà.** Với tối đa 3 offer và thứ tự gần-trước, hầu hết booking đã tìm được người nhận; trần của mọi policy chỉ cao hơn
   Nearest Driver khoảng 1.2 pp. Dư địa thật nằm ở **tốc độ và hiệu suất dispatch** (first-offer acceptance, số offer mỗi booking).
3. **Tối ưu một mục tiêu làm hỏng các mục tiêu khác.** Oracle tối đa hoá xác suất nhận nên ưu tiên tài xế xa hơn nhưng dễ nhận hơn: ETA tăng
   ≈ 17.6%, và vì ETA dài làm tăng khả năng huỷ, cancellation tăng 1.3 pp — kết quả là **số chuyến hoàn thành không tăng**. Đây chính là
   trade-off T1/T6 trong PRD và cảnh báo ở research §2.3, nay có số đo cụ thể.
4. **Weighted rule là đối thủ thật sự.** Nhờ phạt ETA và cancellation lịch sử, rule đặt tay tăng completed rate +1.3 pp và giảm cancellation,
   trong khi ETA chỉ tăng ≈ 6.3% (trong ngưỡng H2).
5. **Validation và test nhất quán** về chiều và độ lớn của mọi chênh lệch.

## 6. Hệ quả & quyết định cần chốt trước Phase 5

Bằng chứng ở §5 cho thấy nếu model ML chỉ xếp hạng theo P(accept) như kế hoạch ban đầu, nó nhiều khả năng lặp lại vấn đề của Oracle ở mức nhẹ
hơn: MSR tăng rất ít, ETA và cancellation xấu đi. PRD Q2 (score cuối là gì) trở thành quyết định quan trọng nhất của Phase 5:

| Phương án | Mô tả | Ưu | Nhược |
|---|---|---|---|
| A. Chỉ P(accept) | Đúng như master prompt | Đơn giản nhất | Có thể vi phạm H2, H3; completed rate khó tăng |
| B. P(accept) có ràng buộc ETA | Chỉ xếp theo P(accept) trong các ứng viên có ETA ≤ ETA nhỏ nhất + Δ phút; Δ chọn trên validation | Vẫn một model; dễ giải thích với Ops | Thêm một tham số; không xử lý trực tiếp cancellation |
| C. Utility | `P(accept) × (1 − P(cancel)) − λ · ETA`; λ chọn trên validation dưới ràng buộc guardrail | Tối ưu đúng completed trips | Cần model thứ hai cho cancellation (nhãn lệch, 13%); cần xác suất được calibrate |

Cũng cần PM quyết định: có nên nâng **completed rate** lên ngang hàng North Star (MSR) không, vì nó phản ánh cả "được nhận" lẫn "không bị huỷ".

> **Cập nhật Phase 5 (2026-09-16).** Cả hai câu hỏi đã được chốt: completed rate trở thành North Star thứ hai
> ([PRD §8.2](product-requirements.md)), và Phase 5 chỉ train **một** model P(accept) — phương án A và B sẽ được so sánh trên
> validation ở Phase 6 ([research §10.7](research.md)). Phương án C (utility có P(cancel)) không được chọn cho MVP.

## 7. Giới hạn

- Kết quả dựa trên simulator; chỉ đúng trong giả định của mô hình hành vi (data/README.md §5, §9).
- Oracle biết cả nhiễu của từng offer, nên là cận trên *lỏng* cho acceptance — nhưng không phải cận trên cho completed rate (§5.3).
- Booking được xử lý độc lập (PRD A4): không có cạnh tranh giữa các booking cho cùng một tài xế.
- Trọng số weighted rule không được tối ưu — một rule được tinh chỉnh có thể còn mạnh hơn.
- Bootstrap coi các booking là độc lập; vì tài xế xuất hiện ở nhiều booking, khoảng tin cậy có thể hơi hẹp hơn thực tế.

## 8. Tái tạo

```bash
python -m ml.data.generate            # dataset mặc định (seed 42)
python -m ml.evaluation.run_baseline  # → ml/evaluation/results/baseline.json + bảng markdown
```
