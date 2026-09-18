# SmartMatch AI — Evaluation

| | |
|---|---|
| Version | 1.0 |
| Phase | 6 — Evaluation & Benchmarking (protocol và baseline: Phase 4) |
| Status | Done — chờ review |
| Last updated | 2026-09-16 |

Tài liệu liên quan: [PRD §8 — Metrics](product-requirements.md) · [Research §6–7, §10](research.md) · [Data](../data/README.md).
Kết quả gốc: [`evaluation.json`](../ml/evaluation/results/evaluation.json) ·
[`baseline.json`](../ml/evaluation/results/baseline.json) · [`ablations.json`](../ml/evaluation/results/ablations.json) ·
[`training.json`](../ml/training/results/training.json).

> Mọi con số trong tài liệu này được sinh bởi các lệnh ở §11 trên dataset mặc định (generator 1.0.0, seed 42).
> Không có số nào được viết tay. Kết quả chỉ đúng trong giả định của simulator (§10).

---

## 1. Tóm tắt Phase 6

Trên **tập test** (13,946 booking, 2026-07-19 → 07-26), so với baseline Nearest Driver:

| Metric | Baseline (Nearest Driver) | ML (policy được chọn) | Improvement |
|---|---|---|---|
| Matching success rate | 85.6% | **86.5%** | **+0.9 pp**, 95% CI [+0.7, +1.1] |
| Acceptance rate (offer-level) | 62.1% | **64.2%** | **+2.1 pp**, 95% CI [+1.8, +2.4] |
| Average ETA của tài xế đã nhận | 7.28 phút | 7.43 phút | +0.14 phút (**+2.0%**), 95% CI [+0.12, +0.17] |
| Cancellation rate | 11.9% | 12.2% | +0.3 pp, 95% CI [+0.0, +0.6] |
| Completed rate | 75.5% | **75.9%** | **+0.5 pp**, 95% CI [+0.2, +0.8] |

Policy được chọn (quyết định trên **validation**, §3): **Logistic Regression + ràng buộc ETA 1 phút** — xếp theo P(accept)
nhưng chỉ trong nhóm ứng viên có ETA không quá 1 phút chậm hơn người nhanh nhất của booking đó (PRD Q2, phương án B).

Bốn kết luận:

1. **ML thắng Nearest Driver ở cả hai North Star** và không vi phạm ngưỡng ETA: MSR +0.9 pp, completed rate +0.5 pp,
   ETA chỉ +2.0% (ngưỡng H2 là +15%). Cả hai khoảng tin cậy đều không chứa 0.
2. **Nhưng weighted rule đặt tay vẫn tốt hơn ML ở đúng metric quan trọng nhất**: completed rate 76.8% vs 75.9%,
   cancellation 11.2% vs 12.2%. ML chỉ thắng rule ở MSR (86.5% vs 86.4%) và tốc độ xác nhận.
3. **H3 không đạt.** Cancellation rate của ML cao hơn baseline +0.34 pp với 95% CI [+0.02, +0.64] — không chứa 0.
   Đây là hệ quả trực tiếp của việc model chỉ tối ưu P(accept) (PRD T6).
4. **Ràng buộc ETA là phần quan trọng nhất của thiết kế policy.** Nếu chỉ xếp theo P(accept) (phương án A), completed rate
   **không** hơn baseline (75.5% vs 75.5%, CI [−0.4, +0.5]) dù MSR và acceptance rate tăng mạnh hơn. Trả lời cho PRD Q2:
   **phương án B**.

---

## 2. Protocol

Không thay đổi so với Phase 4, nên số của baseline và của ML so sánh được trực tiếp.

### 2.1 Dữ liệu & split

- Split theo thời gian ở cấp booking (ADR-011): **train** 2026-06-01 → 07-10 (train model ở Phase 5),
  **validation** 07-11 → 07-18 (14,191 booking) để chọn policy, **test** 07-19 → 07-26 (13,946 booking) để báo cáo.
- Model được train **chỉ trên tập train** (research §10.1), nên validation vẫn là tập sạch để chọn ràng buộc ETA.
- Tập test được đọc **một lần**, sau khi mọi lựa chọn đã chốt trên validation.
- Candidate set: snapshot ứng viên của Phase 2 (tài xế available, đúng loại xe, trong bán kính 5 km). **Mọi policy dùng đúng tập này** —
  chỉ thứ tự offer khác nhau. Việc lọc tài xế trực tiếp từ database thuộc Phase 7–9.

### 2.2 Replay offer tuần tự

1. Mỗi policy cho một score cho từng ứng viên; offer theo score giảm dần, hoà thì theo `driver_id`.
2. Offer lần lượt tối đa **N = 3** lượt, dừng khi có người nhận.
3. Tài xế nhận ⇔ `u_accept < p_accept`; chuyến bị huỷ ⇔ đã nhận và `u_cancel < p_cancel`. Các số ngẫu nhiên `u` cố định cho từng cặp
   (booking, driver) — **common random numbers** — nên mọi policy được đánh giá trong cùng một "thế giới".
4. Policy chỉ nhận các cột có tại thời điểm quyết định ([`ml/features/view.py`](../ml/features/view.py)); không thấy nhãn,
   `offer_rank`, `logging_policy` hay dữ liệu oracle. Riêng Oracle đọc `p_accept` và chỉ tồn tại trong module evaluation.

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
| Weighted rule | Baseline phụ | Điểm theo trọng số đặt tay, **cố định trước khi chạy đánh giá, không tinh chỉnh theo kết quả** |
| ML — Logistic Regression (A) | Model tuyến tính, xếp theo P(accept) | Phase 5, 28 feature |
| ML — XGBoost (A) | Model gradient boosting, xếp theo P(accept) | Phase 5, 28 feature |
| **ML — policy được chọn (B)** | **Đề xuất của Phase 6** | P(accept) trong ràng buộc ETA; model và ngưỡng chọn trên validation (§3) |
| Oracle | Mức trần cho acceptance, **không** phải cho completed rate | Xác suất nhận thật `p_accept` của simulator giảm dần |

Ứng viên không thoả ràng buộc ETA **không bị loại** khỏi danh sách, chỉ bị đẩy xuống sau mọi ứng viên thoả — nếu loại hẳn,
những booking mà tất cả ứng viên "nhanh" đều từ chối sẽ mất cơ hội ghép và MSR sẽ giảm.

Trọng số weighted rule (đơn vị "điểm"; 1 phút ETA = −0.2 điểm):

| Thành phần | Trọng số | Tương đương 1 phút ETA |
|---|---|---|
| `estimated_eta_min` | −0.2 / phút | — |
| `acceptance_rate` | +2.0 | +10 pp acceptance lịch sử |
| `cancellation_rate` | −2.0 | −10 pp cancellation lịch sử |
| `rating` − 4.7 (thiếu rating tính là 4.7) | +0.5 / sao | +0.4 sao |
| `idle_time_min` (chặn ở 30 phút) | +0.02 / phút | +10 phút rảnh |

---

## 3. Chọn policy trên validation

**Quy tắc chọn, cố định trong code trước khi chạy** ([`run_evaluation.py`](../ml/evaluation/run_evaluation.py)):
lấy policy có **completed rate cao nhất** trong số những policy giữ được cả hai guardrail — ETA trung bình không quá **+15%**
so với baseline (H2) và cancellation rate không cao hơn baseline (H3). Nếu không policy nào giữ được H3 thì chỉ ràng buộc H2 và
ghi rõ điều đó vào kết quả.

Mỗi model được chạy với 9 mức ràng buộc ETA; mức ∞ chính là phương án A. Baseline trên validation:
MSR 83.8%, completed rate 73.5%, ETA 7.378 phút (**trần H2: 8.485 phút**), cancellation 12.27%.

| Policy | MSR | Completed rate | Avg ETA (phút) | Cancellation |
|---|---|---|---|---|
| LogReg + ETA ≤ min + 0 phút | 84.8% | 74.12% | 7.51 (+1.8%) | 12.64% |
| LogReg + 0.5 phút | 84.8% | 74.04% | 7.50 (+1.6%) | 12.67% |
| **LogReg + 1 phút** | **84.8%** | **74.12%** | **7.51 (+1.8%)** | **12.57%** |
| LogReg + 2 phút | 84.7% | 73.98% | 7.58 (+2.7%) | 12.62% |
| LogReg + 3 phút | 84.6% | 73.79% | 7.67 (+4.0%) | 12.80% |
| LogReg + 5 phút | 84.6% | 73.39% | 7.86 (+6.6%) | 13.20% |
| LogReg + 8 phút | 84.7% | 73.42% | 8.04 (+8.9%) | 13.36% |
| LogReg + 12 phút | 84.8% | 73.43% | 8.09 (+9.6%) | 13.45% |
| LogReg — chỉ P(accept) (A) | 84.9% | 73.45% | 8.09 (+9.7%) | 13.45% |
| XGBoost + 0 phút | 84.9% | 74.12% | 7.53 (+2.1%) | 12.66% |
| XGBoost + 0.5 phút | 84.8% | 74.04% | 7.52 (+1.9%) | 12.72% |
| XGBoost + 1 phút | 84.8% | 74.10% | 7.53 (+2.1%) | 12.64% |
| XGBoost + 2 phút | 84.7% | 73.94% | 7.60 (+3.1%) | 12.72% |
| XGBoost + 3 phút | 84.7% | 73.79% | 7.71 (+4.5%) | 12.87% |
| XGBoost + 5 phút | 84.6% | 73.42% | 7.94 (+7.6%) | 13.19% |
| XGBoost + 8 phút | 84.7% | 73.26% | 8.16 (+10.7%) | 13.49% |
| XGBoost + 12 phút | 84.9% | 73.29% | 8.25 (+11.8%) | 13.63% |
| XGBoost — chỉ P(accept) (A) | 84.9% | 73.29% | 8.26 (+12.0%) | 13.66% |

Ba điều rút ra ngay từ bảng này:

- **Không policy ML nào giữ được H3** (mọi cancellation rate đều > 12.27%) → quy tắc chọn rơi vào nhánh "chỉ ràng buộc H2",
  và điều đó được ghi lại trong `evaluation.json`. Đây là cảnh báo sớm cho kết quả trên test.
- **Ràng buộc ETA chặt (0–2 phút) cho completed rate cao nhất** ở cả hai model, nới càng rộng completed rate càng giảm —
  đúng cơ chế Phase 4 đã chỉ ra: ETA dài hơn kéo theo huỷ nhiều hơn.
- Ba lựa chọn tốt nhất (LogReg 0 phút, LogReg 1 phút, XGBoost 0 phút) chênh nhau **dưới 0.01 pp** completed rate. Việc chọn
  LogReg + 1 phút thay vì hai cái kia nằm trong vùng nhiễu, không phải bằng chứng nó tốt hơn (§10).

---

## 4. Kết quả trên tập test

13,946 booking (13,788 có ứng viên), 111,046 dòng ứng viên. Trong ngoặc: chênh lệch so với Nearest Driver, 95% CI.

| Metric | Random | **Nearest Driver** | Weighted rule | ML LogReg (A) | ML XGBoost (A) | **ML chọn (B)** | Oracle |
|---|---|---|---|---|---|---|---|
| Matching success rate | 73.5% ([−12.8, −11.6] pp) | **85.6%** | 86.4% ([+0.6, +1.0] pp) | 86.6% ([+0.7, +1.2] pp) | 86.6% ([+0.8, +1.2] pp) | **86.5% ([+0.7, +1.1] pp)** | 86.8% ([+1.0, +1.5] pp) |
| Acceptance rate | 39.4% ([−23.4, −21.9] pp) | **62.1%** | 65.4% ([+2.9, +3.7] pp) | 66.3% ([+3.8, +4.7] pp) | 66.6% ([+4.0, +4.9] pp) | **64.2% ([+1.8, +2.4] pp)** | 68.0% ([+5.4, +6.3] pp) |
| Average ETA, matched (phút) | 10.75 ([+3.37, +3.55]) | **7.28** | 7.75 ([+0.43, +0.49]) | 8.01 ([+0.69, +0.76]) | 8.19 ([+0.87, +0.94]) | **7.43 ([+0.12, +0.17])** | 8.57 ([+1.24, +1.33]) |
| P90 ETA, matched (phút) | 16.83 | **12.65** | 13.19 | 13.52 | 13.64 | **12.96** | 14.08 |
| Cancellation rate | 16.1% ([+3.5, +5.0] pp) | **11.9%** | 11.2% ([−1.1, −0.3] pp) | 12.8% ([+0.5, +1.4] pp) | 12.8% ([+0.5, +1.4] pp) | **12.2% ([+0.0, +0.6] pp)** | 13.2% ([+0.9, +1.9] pp) |
| Completed rate | 61.6% ([−14.6, −13.1] pp) | **75.5%** | 76.8% ([+0.9, +1.7] pp) | 75.5% ([−0.4, +0.5] pp) | 75.5% ([−0.4, +0.5] pp) | **75.9% ([+0.2, +0.8] pp)** | 75.3% ([−0.6, +0.4] pp) |
| First-offer acceptance rate | 42.7% ([−30.4, −28.6] pp) | **72.2%** | 76.5% ([+3.8, +4.8] pp) | 77.6% ([+4.8, +6.0] pp) | 77.9% ([+5.0, +6.2] pp) | **73.9% ([+1.3, +2.0] pp)** | 79.6% ([+6.8, +8.0] pp) |
| Offers per booking | 1.885 ([+0.476, +0.505]) | **1.395** | 1.336 ([−0.065, −0.052]) | 1.320 ([−0.082, −0.067]) | 1.316 ([−0.086, −0.071]) | **1.362 ([−0.037, −0.027])** | 1.293 ([−0.110, −0.094]) |

### 4.1 Gap closed

Phần dư địa cải thiện mà mỗi policy lấy được: (policy − baseline) / (oracle − baseline).

| Metric | Weighted rule | ML LogReg (A) | ML XGBoost (A) | ML chọn (B) |
|---|---|---|---|---|
| Matching success rate | 65.7% | 77.9% | **81.4%** | 72.1% |
| Acceptance rate | 56.6% | 72.3% | **76.0%** | 36.2% |
| Completed rate | — | — | — | — |

Completed rate không có dòng vì **oracle không phải cận trên** cho metric này: oracle đạt 75.3%, *thấp hơn* baseline 75.5%
(xếp theo xác suất nhận thật dẫn tới tài xế xa hơn → huỷ nhiều hơn). Tỷ số sẽ vô nghĩa nên code trả về `null`.

---

## 5. ML metrics — tầng classification

Đo trên **20,591 offer có nhãn** của tập test (13,788 booking, acceptance 0.564). Đây là lần đọc nhãn của tập test duy nhất.

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.8833 | **0.8847** |
| PR-AUC | 0.8972 | **0.8986** |
| Log loss | 0.4201 | **0.4178** |
| Brier | 0.1350 | **0.1341** |
| Precision @ 0.5 | **0.8098** | 0.8094 |
| Recall @ 0.5 | 0.8632 | **0.8671** |
| F1 @ 0.5 | 0.8357 | **0.8373** |
| Hit@1 trong booking | 0.6372 | **0.6509** |
| NDCG@5 trong booking | 0.8597 | **0.8645** |

- Precision / Recall / F1 lấy ngưỡng 0.5 và **chỉ mang tính chẩn đoán**: ranking không dùng ngưỡng, chỉ dùng thứ tự điểm.
- Test cao hơn validation một chút (ROC-AUC 0.8847 vs 0.8818) — bình thường, hai tuần khác nhau; không có dấu hiệu overfit.
- Hit@1 và NDCG@5 chỉ tính trên 2,475 booking có ≥ 2 offer có nhãn và ≥ 1 lượt nhận (§10).
- XGBoost thắng Logistic Regression ở hầu hết metric ML, nhưng chênh lệch **không chuyển thành lợi thế business** (§4).

---

## 6. Ablation — feature nào thực sự đóng góp?

Mỗi dòng đổi **đúng một** thứ: tập feature. Cùng tập train, cùng hyperparameter (bộ Phase 5 chọn), cùng early stopping,
cùng protocol replay; đo trên **validation**, xếp hạng theo phương án A.
Nguồn: [`ablations.json`](../ml/evaluation/results/ablations.json).

| Ablation | Feature | ROC-AUC | Chênh AUC | Hit@1 | MSR | Completed rate | Avg ETA |
|---|---|---|---|---|---|---|---|
| Đầy đủ (mốc) | 28 | 0.8818 | — | 0.6213 | 84.9% | 73.29% | 8.26 |
| **AB2** — bỏ lịch sử tài xế + idle time | 22 | 0.8428 | **−0.0390** | **0.3075** | 84.0% | 73.73% | 7.35 |
| **AB3** — bỏ traffic / weather / time | 15 | 0.8792 | −0.0026 | 0.6138 | 84.9% | 73.38% | 8.28 |
| **AB4** — chỉ distance + ETA | 2 | 0.8319 | −0.0499 | 0.3584 | 84.0% | 73.74% | 7.37 |
| **AB5** — bỏ relative feature | 27 | 0.8820 | +0.0002 | 0.6233 | 84.9% | 73.34% | 8.28 |
| **AB1** — Logistic Regression (mọi feature) | 28 | 0.8810 | −0.0008 | 0.6197 | 84.9% | 73.45% | 8.09 |

1. **Lịch sử tài xế là nhóm feature quan trọng nhất** — bỏ đi làm AUC giảm 0.039 và Hit@1 **sụp từ 0.62 xuống 0.31**.
   Đúng dự đoán của EDA (§9.8.8): traffic / weather / time là tín hiệu chung của cả booking nên không giúp phân biệt
   *giữa các tài xế trong cùng một booking*; chỉ feature riêng của từng tài xế mới làm được.
2. **Context gần như không đóng góp** (AB3: −0.0026 AUC) vì tác động của nó đã đi qua ETA (EDA §9.5).
3. **Relative feature dư thừa** (AB5: +0.0002) — trùng tín hiệu với `distance_km`, khớp với ρ = 0.97 ở EDA §9.7.
   Có thể bỏ để đơn giản hoá ở Phase 7.
4. **Model càng yếu càng giống baseline**: AB2 và AB4 không có feature tài xế nên xếp hạng gần như theo khoảng cách —
   ETA tụt về 7.35–7.37 phút (baseline 7.38) và completed rate *cao hơn* model đầy đủ. Một lời nhắc rằng completed rate cao
   không đồng nghĩa với model tốt hơn, nếu nó đạt được bằng cách bắt chước baseline.

---

## 7. Diễn giải

1. **ML học được thứ Nearest Driver không thấy.** Acceptance rate +2.1 đến +4.5 pp, first-offer acceptance +1.3 đến +6.2 pp,
   số offer mỗi booking giảm — đúng dư địa Phase 4 đã chỉ ra. Về MSR, ML lấy được **72–81% khoảng cách tới oracle**.
2. **Tối ưu P(accept) đơn thuần không tạo ra chuyến hoàn thành.** Phương án A tăng acceptance mạnh nhất nhưng completed rate
   đứng yên (75.5% vs 75.5%): đúng hiện tượng Oracle của Phase 4, ở mức nhẹ hơn. Ràng buộc ETA 1 phút hy sinh hơn một nửa
   phần acceptance tăng thêm để đổi lấy +0.5 pp completed rate — đánh đổi đáng giá theo North Star mới.
3. **Weighted rule vẫn là đối thủ mạnh nhất ở completed rate** (76.8% vs 75.9%) và là policy duy nhất **giảm** cancellation
   (−0.7 pp). Lý do rõ ràng: nó phạt trực tiếp `cancellation_rate` lịch sử của tài xế, còn model ML không hề được cho biết
   mục tiêu đó tồn tại. Đây không phải lỗi của ML mà là hệ quả của việc chọn target (PRD T6).
4. **Đường đi tiếp theo đã rõ và có số đo hậu thuẫn**: phương án C (utility `P(accept) × (1 − P(cancel)) − λ·ETA`) là cách
   trực tiếp để lấy lại phần weighted rule đang thắng. Hoãn nó ở Phase 5 là quyết định có ý thức; Phase 6 cho biết chi phí
   của quyết định đó: khoảng 0.9 pp completed rate và 1.0 pp cancellation.
5. **Phi tuyến vẫn không đáng giá.** XGBoost hơn Logistic Regression ở hầu hết metric ML nhưng thua ở completed rate trên
   validation, và chính LogReg được chọn. Với dữ liệu này model tuyến tính đã đủ — có lợi cho latency và giải thích ở Phase 7.

---

## 8. Giả thuyết (PRD §8.3) — kết luận trên tập test

| ID | Giả thuyết | Kết luận | Bằng chứng |
|---|---|---|---|
| H1 | ML đạt MSR cao hơn Nearest Driver | **Đạt** | +0.89 pp, 95% CI [+0.67, +1.09] |
| H1b | ML đạt completed rate cao hơn Nearest Driver | **Đạt** | +0.49 pp, 95% CI [+0.16, +0.81] |
| H2 | Average ETA không tăng quá +15% | **Đạt** | +2.0% (7.28 → 7.43 phút) |
| H3 | Cancellation rate không cao hơn baseline | **Không đạt** | +0.34 pp, 95% CI [+0.02, +0.64] — không chứa 0 |
| H4 | ROC-AUC tốt hơn hẳn 0.5 nhưng không phi thực tế | **Đạt** | 0.8833 (trần oracle 0.902) |

H3 chỉ bị coi là không đạt khi khoảng tin cậy nằm **hoàn toàn trên 0**, tức là có bằng chứng cancellation thực sự cao hơn.
Kết quả được báo cáo đúng như đo được, theo yêu cầu của PRD §8.3.

---

## 9. Kết quả baseline trên validation (tham chiếu, Phase 4)

Số của Nearest Driver, Weighted rule, Random và Oracle trên **test** nằm ở §4. Bảng dưới là tập validation, dùng để phát triển
model ở Phase 5 mà không chạm vào test. 14,191 booking (13,977 có ứng viên).

| Metric | Random | **Nearest Driver** | Weighted rule | Oracle |
|---|---|---|---|---|
| Matching success rate | 71.1% ([−13.3, −12.1] pp) | **83.8%** | 84.6% ([+0.6, +1.1] pp) | 85.2% ([+1.2, +1.7] pp) |
| Acceptance rate | 37.8% ([−22.6, −21.2] pp) | **59.6%** | 63.0% ([+3.0, +3.8] pp) | 65.7% ([+5.5, +6.5] pp) |
| Average ETA, matched (phút) | 10.90 ([+3.43, +3.61]) | **7.38** | 7.83 ([+0.42, +0.48]) | 8.62 ([+1.20, +1.29]) |
| P90 ETA, matched (phút) | 17.03 | **12.78** | 13.36 | 14.30 |
| Cancellation rate | 16.6% ([+3.6, +5.1] pp) | **12.3%** | 11.8% ([−0.8, −0.0] pp) | 13.8% ([+0.9, +2.0] pp) |
| Completed rate | 59.3% ([−15.0, −13.5] pp) | **73.5%** | 74.6% ([+0.7, +1.5] pp) | 73.5% ([−0.5, +0.5] pp) |

---

## 10. Giới hạn

- Kết quả dựa trên simulator; chỉ đúng trong giả định của mô hình hành vi (data/README.md §5, §9).
- **Oracle không phải cận trên cho completed rate** (§4.1), chỉ cho acceptance và MSR.
- **Lựa chọn trên validation nằm trong vùng nhiễu**: ba policy tốt nhất chênh nhau < 0.01 pp completed rate (§3).
  Kết luận "phương án B tốt hơn A" thì chắc chắn (chênh ≈ 0.8 pp); kết luận "ngưỡng 1 phút tốt hơn 0 phút" thì không.
- Chọn policy trên validation bằng point estimate, không bootstrap → có thể lạc quan nhẹ khi mang sang test.
- Hit@1 / NDCG@5 chỉ đo được trên 2,475 / 13,788 booking có ≥ 2 offer có nhãn và ≥ 1 lượt nhận; các ứng viên có nhãn do
  logging policy chọn, không phải mẫu ngẫu nhiên. Thước đo thứ hạng đáng tin là replay ở §4.
- Trọng số weighted rule không được tối ưu — một rule được tinh chỉnh có thể còn mạnh hơn nữa.
- Bootstrap coi các booking là độc lập; vì tài xế xuất hiện ở nhiều booking, khoảng tin cậy có thể hơi hẹp hơn thực tế.
- Booking được xử lý độc lập (PRD A4): không có cạnh tranh giữa các booking cho cùng một tài xế.

---

## 11. Tái tạo

```bash
python -m ml.data.generate              # dataset mặc định (seed 42)
python -m ml.training.train             # Phase 5: model + ml/training/results/training.json
python -m ml.evaluation.run_baseline    # Phase 4: baseline.json
python -m ml.evaluation.run_evaluation  # Phase 6: evaluation.json (chọn trên validation, đo trên test)
python -m ml.evaluation.run_ablations   # Phase 6: ablations.json
```

Cả ba file JSON tái tạo **byte-identical** giữa hai lần chạy (kiểm bằng SHA-256).
