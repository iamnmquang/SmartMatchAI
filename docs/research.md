# SmartMatch AI — Research

| | |
|---|---|
| Version | 0.2 |
| Phase | 1 — Research; §9 EDA findings: Phase 3 |
| Status | Done — chờ review |
| Last updated | 2026-09-15 |

Tài liệu liên quan: [Product Requirements](product-requirements.md) · [Architecture](architecture.md).

> **Phạm vi.** Tài liệu này tổng hợp các hướng tiếp cận bài toán ghép tài xế – booking và chọn baseline cùng ML approach cho SmartMatch.
> Một số paper được trích dẫn do nhân viên các nền tảng gọi xe công bố. Chúng chỉ được dùng như **related work học thuật công khai**;
> SmartMatch không giả định kiến trúc hay logic production của bất kỳ tổ chức nào.

---

## 1. Tóm tắt

| | Quyết định |
|---|---|
| **Baseline** | **Nearest Driver** — cùng candidate generation với ML, chỉ khác cách xếp hạng. |
| **ML approach** | **Pointwise binary classification**: ước lượng P(accept) cho từng cặp (booking, driver) bằng **XGBoost**, sắp xếp giảm dần, trả Top-K. Logistic Regression làm mốc tuyến tính. |
| **Explanation** | TreeSHAP có sẵn trong XGBoost (`pred_contribs`) → reason dạng template. |
| **Kiến trúc ranking** | Rule-based candidate filter → ML scoring → sort → Top-K. |
| **Không dùng cho MVP** | Learning-to-Rank (giữ làm ablation), optimization-based matching (future work), deep learning, reinforcement learning. |

Lý do cốt lõi (§2.3): dưới giả định offer tuần tự và tài xế quyết định độc lập, **sắp xếp theo xác suất nhận chuyến thật là tối ưu**
cho Matching Success Rate, số offer kỳ vọng và Acceptance Rate. Vì vậy bài toán ML đúng là *ước lượng xác suất đó đủ tốt để xếp đúng thứ tự*.

---

## 2. Problem formulation

### 2.1 Ký hiệu

- Booking $b$ với context $c_b$ (thời điểm, traffic, weather).
- Candidate set $D_b$: các tài xế available trong bán kính $R$, loại xe phù hợp.
- Feature vector $x_{b,d}$ cho mỗi cặp booking – driver.
- Outcome khi được offer: $y_{b,d} \in \{0, 1\}$ (accepted). Nếu đã nhận: $z_{b,d} \in \{0, 1\}$ (cancelled).
- Xác suất thật (chỉ simulator biết): $p_{b,d} = P(y_{b,d} = 1)$.
- Policy $\pi$ tạo thứ tự $\sigma = (d_1, d_2, \dots)$ trên $D_b$. Offer được gửi tuần tự, tối đa $N$ lượt, dừng khi có người nhận.

### 2.2 Objective

Nếu các tài xế quyết định độc lập:

$$\text{MSR}_b(\sigma) = 1 - \prod_{k=1}^{N} \left(1 - p_{b,d_k}\right)$$

$$\mathbb{E}\left[\text{offers}_b(\sigma)\right] = \sum_{k=1}^{N} \prod_{i<k} \left(1 - p_{b,d_i}\right)$$

Acceptance Rate (offer-level) trên toàn tập = tổng booking matched / tổng offer đã gửi.

### 2.3 Kết quả quan trọng: sort theo xác suất là tối ưu

**Mệnh đề.** Giả sử (i) tài xế quyết định độc lập, (ii) $p$ không phụ thuộc vào thứ tự offer, (iii) mục tiêu chỉ là ghép thành công.
Khi đó thứ tự sắp theo $p$ giảm dần đồng thời:

1. **Tối đa MSR.** MSR chỉ phụ thuộc vào *tập* $N$ tài xế đầu tiên → chọn $N$ người có $p$ lớn nhất.
2. **Tối thiểu số offer kỳ vọng.** Đổi chỗ hai vị trí liền kề $k$ và $k+1$ chỉ làm thay đổi hạng tử thứ $k+1$, bằng
   $\prod_{i<k}(1-p_{d_i}) \cdot (1-p_{d_k})$. Hạng tử này nhỏ hơn khi $p_{d_k}$ lớn hơn → người có $p$ cao nên đứng trước (exchange argument).
3. **Tối đa Acceptance Rate.** Mỗi booking vừa có tử số (matched) lớn nhất vừa có mẫu số (offers) nhỏ nhất.

**Kiểm chứng trong Phase 1** (script brute-force, chỉ dùng thư viện chuẩn Python):
2.000 bộ xác suất ngẫu nhiên × 720 hoán vị (6 candidates, $N = 3$) → **0 hoán vị** tốt hơn thứ tự giảm dần theo MSR, số offer kỳ vọng hoặc AR.
Công thức ở §2.2 khớp với mô phỏng Monte Carlo 200.000 lần.

Ví dụ minh hoạ toán học với $p = [0.35, 0.8, 0.2, 0.6, 0.5]$, $N = 3$. Đây **không** phải kết quả thực nghiệm của SmartMatch.

| Thứ tự offer | MSR | E[offers] | AR |
|---|---|---|---|
| Giảm dần theo p | 0.960 | 1.28 | 0.750 |
| Theo thứ tự cho sẵn | 0.896 | 1.78 | 0.503 |
| Tăng dần theo p | 0.740 | 2.32 | 0.319 |

**Hệ quả cho thiết kế:**

- Bài toán ML đúng là **ước lượng $p$ để xếp đúng thứ tự trong từng booking** → pointwise classification phù hợp tự nhiên.
- Nếu mục tiêu chỉ là MSR, **chỉ cần thứ tự đúng**, không cần xác suất được calibrate.
- **Khi thêm chi phí ETA, kết luận trên không còn đúng.** Với utility
  $U = \sum_k P(\text{match tại lượt } k) \cdot (V - c \cdot \text{ETA}_k)$, $V = 10$, $c = 0.5$:
  tài xế A ($p = 0.60$, ETA 12 phút) và B ($p = 0.55$, ETA 3 phút). Offer A trước → $U = 4.270$; B trước → $U = 5.755$.
  Người có $p$ thấp hơn một chút nhưng gần hơn nhiều lại nên đứng trước.
  → Khi kết hợp nhiều mục tiêu (PRD Q2), cần **xác suất được calibrate** và một **trọng số business** $c$. Quyết định ở Phase 5–6.

### 2.4 Các giả định có thể bị vi phạm

| Giả định | Thực tế có thể khác | Xử lý |
|---|---|---|
| Tài xế quyết định độc lập | Tài xế cùng khu vực chịu cùng điều kiện | Chấp nhận có điều kiện trên feature context |
| $p$ không phụ thuộc thứ tự offer | Mỗi lượt từ chối làm hành khách chờ thêm, có thể huỷ | Theo dõi E[offers] như chỉ số phụ; mở rộng simulator nếu cần |
| Booking độc lập với nhau | Nhiều booking tranh cùng một tài xế | Ngoài MVP — xem §3.6 |

---

## 3. Related work: sáu hướng tiếp cận

### 3.1 Rule-based matching

- **Cách hoạt động.** Hai lớp:
  (1) *hard constraints* — available, trong bán kính $R$, loại xe phù hợp, rating ≥ ngưỡng;
  (2) *điểm có trọng số chọn tay*, ví dụ `score = −w1·ETA + w2·rating + w3·acceptance_rate − w4·cancellation_rate + w5·idle_time`.
- **Ưu điểm.** Không cần dữ liệu có nhãn. Minh bạch, dễ audit. Encode trực tiếp policy business hoặc quy định. Deterministic. Triển khai ngay.
- **Nhược điểm.** Trọng số dựa trên trực giác; không có cách "đúng" để chọn nếu không thử nghiệm. Không bắt được quan hệ phi tuyến và tương tác
  giữa các yếu tố. Số rule tăng nhanh, khó bảo trì. Không tự thích nghi khi hành vi thay đổi.
- **Khi nào nên dùng.** Cold start khi chưa có dữ liệu. Ràng buộc bắt buộc (eligibility, an toàn, pháp lý) — lớp này **nên luôn tồn tại**,
  kể cả khi đã có ML. Khi yêu cầu giải thích tuyệt đối.
- **Dữ liệu & feature.** Thuộc tính tài xế, vị trí, loại xe, trạng thái. Không cần outcome.
- **Evaluation metrics.** Chỉ business metrics (MSR, AR, Average ETA, CR) qua thử nghiệm hoặc simulation.
- **Vai trò trong SmartMatch.** **Candidate filter** trước ML. Có thể dùng làm **baseline phụ** (weighted rule score) để tránh so ML
  với một baseline quá yếu — PRD Q9.

### 3.2 Nearest-driver matching

- **Cách hoạt động.** Với mỗi booking, chọn tài xế available có khoảng cách (hoặc ETA) nhỏ nhất; nếu bị từ chối thì chuyển sang người gần tiếp theo.
  Biến thể: khoảng cách đường chim bay vs ETA theo mạng đường thật; xử lý booking theo thứ tự đến (first-come-first-served).
- **Ưu điểm.** Đơn giản nhất, không cần train. Rất nhanh (có thể dùng spatial index). Tối thiểu trực tiếp ETA đón cho từng booking. Dễ hiểu, cảm giác công bằng.
- **Nhược điểm.** Bỏ qua khả năng nhận chuyến → nhiều lượt từ chối làm hành khách chờ thêm. Bỏ qua chất lượng và hành vi tài xế.
  Greedy/myopic: tốt cho từng booking nhưng có thể kém toàn cục khi nhiều booking tranh cùng tài xế [Zhang et al., 2017; Yan et al., 2020].
  Khoảng cách đường chim bay khác ETA thực.
- **Khi nào nên dùng.** Làm baseline. Khi tỷ lệ nhận chuyến gần như luôn cao (ví dụ auto-accept). Mật độ thấp, ít tranh chấp.
- **Dữ liệu & feature.** Vị trí tài xế và booking, trạng thái available, (ETA từ routing engine nếu có).
- **Evaluation metrics.** Average pickup ETA/distance, MSR, AR, CR.
- **Vai trò trong SmartMatch.** **Baseline chính.** Trong simulator (Phase 2), traffic là context cấp booking nhưng ETA có thêm
  detour factor ngẫu nhiên theo từng cặp booking – driver, nên thứ tự theo distance *gần* trùng — nhưng không hoàn toàn trùng — với thứ tự theo ETA.

### 3.3 Classification (pointwise)

- **Cách hoạt động.** Học $\hat{p} = f(x_{b,d})$ bằng binary classifier trên từng cặp (booking, driver) đã được offer, rồi xếp hạng theo $\hat{p}$.
  Model thường dùng: Logistic Regression, Random Forest, gradient boosting (XGBoost [Chen & Guestrin, 2016], LightGBM [Ke et al., 2017]), neural network.
- **Ưu điểm.** Framing đơn giản; tooling và metric chuẩn. Mỗi offer là một dòng dữ liệu → dùng được cả khi mỗi booking chỉ có một người được offer.
  Output là xác suất → dùng lại được cho utility, optimization, analytics. Giải thích được bằng SHAP [Lundberg & Lee, 2017].
- **Nhược điểm.** Loss tính trên từng dòng, không trực tiếp tối ưu thứ tự trong booking: model có thể tốn năng lực học hiệu ứng cấp booking
  (ví dụ mưa làm giảm $p$ của mọi tài xế) vốn không giúp xếp hạng. ROC-AUC toàn cục trộn cả so sánh giữa các booking khác nhau → cần thêm metric theo nhóm.
  Nếu dùng $\hat{p}$ như một giá trị (utility), cần calibration [Niculescu-Mizil & Caruana, 2005].
  **Selection bias**: nhãn chỉ có cho tài xế đã được offer theo policy cũ → model yếu ở vùng mà policy cũ ít khám phá. Class imbalance.
- **Khi nào nên dùng.** Dữ liệu tabular, nhãn nhị phân theo từng cặp, cần xác suất, cần giải thích.
- **Dữ liệu & feature.** Pair (distance, ETA); lịch sử tài xế (rating, acceptance rate, cancellation rate, completed trips, idle time);
  booking (quãng đường chuyến, passenger type); context (giờ, ngày, thời tiết, traffic). Chi tiết ở §5.2.
- **Evaluation metrics.** ROC-AUC, PR-AUC, log loss, Brier score / reliability curve; Precision/Recall/F1 tại một threshold;
  thêm ranking metrics theo booking và business metrics qua simulation.
- **Vai trò trong SmartMatch.** **Hướng được chọn** (§6.2).

### 3.4 Ranking (score-and-sort, multi-stage)

- **Cách hoạt động.** "Ranking" ở đây là *cách tổ chức hệ thống*: candidate generation (rẻ, ưu tiên recall) → scoring (chính xác hơn) →
  re-ranking theo business rule hoặc nhiều mục tiêu → Top-K. Đây là mẫu phổ biến trong recommender systems [Covington et al., 2016].
  Bộ chấm điểm có thể là rule, pointwise model hoặc LTR. Điểm đa mục tiêu có dạng utility, ví dụ
  $s = \hat{p}_{accept} \cdot (1 - \hat{p}_{cancel}) - \lambda \cdot \text{ETA}$.
- **Ưu điểm.** Scale tốt vì phần chấm điểm đắt chỉ chạy trên tập nhỏ. Tách policy (lọc, ràng buộc) khỏi phần học.
  Output Top-K hợp với UI cho Ops (human-in-the-loop). Kết hợp được nhiều mục tiêu.
- **Nhược điểm.** Tài xế bị loại ở bước candidate generation không thể được cứu lại (mất recall). Trọng số đa mục tiêu là quyết định business,
  cần thử nghiệm. Có khoảng cách giữa kết quả offline và online.
- **Khi nào nên dùng.** Mỗi request có nhiều candidate; output là danh sách; có nhiều mục tiêu cần cân bằng.
- **Dữ liệu & feature.** Như classification, cộng thêm dữ liệu để đánh giá từng stage (ví dụ candidate tốt nhất có lọt qua bước lọc không).
- **Evaluation metrics.** Recall@N cho candidate generation; NDCG@K [Järvelin & Kekäläinen, 2002], MRR, Hit@K cho ranking; business metrics.
- **Vai trò trong SmartMatch.** **Kiến trúc được áp dụng**: rule-based candidate filter → ML scoring → sort → Top-K + reason.

### 3.5 Learning-to-Rank (LTR)

- **Cách hoạt động.** Học trực tiếp thứ tự trong mỗi nhóm (query = booking). Ba họ phương pháp [Liu, 2009]:
  *pointwise*; *pairwise* — RankSVM [Joachims, 2002], RankNet; *listwise* — ListNet [Cao et al., 2007], LambdaRank/LambdaMART [Burges, 2010],
  trong đó LambdaMART tối ưu gần đúng NDCG. XGBoost hỗ trợ `rank:ndcg` (LambdaMART, mặc định), `rank:pairwise`, `rank:map`;
  nhóm được khai báo qua `qid`; có `lambdarank_unbiased` để giảm position bias.
- **Ưu điểm.** Tối ưu đúng thứ tự trong booking, bỏ qua hiệu ứng chung cấp booking. Khớp với metric ranking (NDCG). Rất thành công trong search.
- **Nhược điểm.** Cần **nhiều candidate có nhãn trong cùng một booking** — log dispatch thực tế thường chỉ có nhãn cho một vài tài xế được offer;
  nhóm chỉ có một phần tử không cho tín hiệu pairwise. **Score không phải xác suất**: không calibrate, không so sánh được giữa các booking
  → không dùng được cho utility có ETA, cho threshold hay làm trọng số optimization. Feedback bị position bias [Joachims et al., 2017].
  Khó giải thích với stakeholder hơn. Thêm nhiều quyết định thiết kế (cách tạo nhóm, gain, truncation level).
- **Khi nào nên dùng.** Mỗi query có nhiều item được đánh giá (search, recommender có impression log); chỉ thứ tự quan trọng; không cần xác suất.
- **Dữ liệu & feature.** Như classification, cộng `qid` và nhãn relevance cho nhiều item trong mỗi nhóm.
- **Evaluation metrics.** NDCG@K, MAP, MRR.
- **Vai trò trong SmartMatch.** **Không dùng cho MVP.** Là **ablation** ở Phase 5–6: `XGBRanker` vs `XGBClassifier` trên cùng feature,
  so NDCG@5 và business metrics. Chỉ có ý nghĩa nếu dữ liệu có nhãn cho nhiều candidate trong mỗi booking (PRD Q8).

### 3.6 Optimization-based matching

- **Cách hoạt động.** Gom các booking trong một cửa sổ thời gian ngắn (batch), dựng đồ thị hai phía booking × tài xế với trọng số cạnh $w_{b,d}$
  (−ETA, $\hat{p}$ hoặc utility), rồi giải bài toán gán: Hungarian method [Kuhn, 1955] (ví dụ SciPy `linear_sum_assignment`), min-cost flow,
  hoặc integer programming có ràng buộc. Phiên bản online: online bipartite matching [Karp et al., 1990].
  Trong ride-hailing: Zhang et al. (2017) dùng tối ưu tổ hợp để tối đa global success rate khi phát đơn;
  Xu et al. (2018) kết hợp learning và planning, tính cả giá trị tương lai của việc gán; Alonso-Mora et al. (2017) giải bài toán gán cho ride-sharing sức chứa lớn;
  Qin et al. (2020) mô tả hướng phát triển từ tối ưu tổ hợp sang semi-MDP và reinforcement learning.
  Tổng quan: Wang & Yang (2019), Yan et al. (2020).
- **Ưu điểm.** Tối ưu toàn cục, tránh xung đột của greedy khi nhiều booking tranh cùng tài xế. Encode được ràng buộc (năng lực, công bằng).
  Kết hợp tự nhiên với ML: ML dự đoán trọng số cạnh, optimizer quyết định phép gán.
- **Nhược điểm.** Batching thêm độ trễ. Chi phí tính toán tăng (Hungarian $O(n^3)$). Cần **simulator động**: trạng thái tài xế theo thời gian,
  thời gian chuyến, cung – cầu đồng thời — phức tạp hơn nhiều. Output gán 1–1 không khớp tự nhiên với use case Top-K cho Ops.
  Khó giải thích từng quyết định vì kết quả của một booking phụ thuộc vào các booking khác.
- **Khi nào nên dùng.** Mật độ cao, nhiều request đồng thời, supply khan hiếm; dispatch ở cấp nền tảng.
- **Dữ liệu & feature.** Vị trí và trạng thái tài xế theo thời gian, luồng booking theo thời gian, trọng số cạnh (ETA, xác suất).
- **Evaluation metrics.** Tổng hoặc trung bình ETA toàn cục, global MSR, tổng utility, driver utilization, solver runtime, độ trễ do batching.
- **Vai trò trong SmartMatch.** **Future work.** MVP giả định các booking độc lập (PRD A4). Chọn pointwise probability giữ ngỏ hướng này:
  $\hat{p}$ dùng được làm trọng số cạnh, còn score của LTR thì không.

---

## 4. So sánh tổng hợp

| Tiêu chí | Rule-based | Nearest driver | Classification | Ranking (multi-stage) | Learning-to-Rank | Optimization |
|---|---|---|---|---|---|---|
| Cần dữ liệu có nhãn | Không | Không | Có | Tuỳ bộ chấm điểm | Có, nhiều item mỗi nhóm | Tuỳ nguồn trọng số |
| Output | Điểm | Thứ tự theo khoảng cách | Xác suất | Top-K | Score tương đối | Phép gán 1–1 |
| Bắt quan hệ phi tuyến | Không | Không | Có | Tuỳ bộ chấm điểm | Có | Tuỳ trọng số |
| Xử lý xung đột giữa booking | Không | Không | Không | Không | Không | **Có** |
| Mức độ dễ giải thích | Rất dễ | Rất dễ | Dễ (SHAP) | Trung bình | Khó hơn | Khó |
| Độ phức tạp triển khai | Thấp | Rất thấp | Trung bình | Trung bình | Trung bình – cao | Cao |
| **Vai trò trong SmartMatch** | Candidate filter; baseline phụ (tuỳ chọn) | **Baseline** | **ML approach** | **Kiến trúc** | Ablation | Future work |

---

## 5. Dữ liệu & feature

### 5.1 Dữ liệu public

Dữ liệu trip công khai như NYC TLC Trip Record Data có thời gian, vị trí đón/trả, quãng đường và cước — nhưng chỉ ghi nhận **chuyến đã diễn ra**,
không có quyết định dispatch, offer hay lượt từ chối của tài xế. Trong phạm vi tìm hiểu ở Phase 1, chưa tìm thấy dataset công khai có
candidate set kèm nhãn nhận/từ chối ở cấp tài xế. → Dùng **synthetic data** (đúng với PRD). Việc tham khảo phân phối thời gian/không gian
từ dữ liệu public là tuỳ chọn, quyết định ở Phase 2 (nếu dùng phải ghi nguồn và điều khoản sử dụng trong `data/README.md`).

### 5.2 Feature taxonomy (ứng viên cho Phase 5 — sẽ được EDA và ablation kiểm chứng)

| Nhóm | Feature | Ghi chú |
|---|---|---|
| Pair | `distance_km`, `eta_min` | Tính tại thời điểm booking |
| Driver — lịch sử | `rating`, `acceptance_rate`, `cancellation_rate`, `completed_trips`, `idle_time_min` | Phải là giá trị **trước** thời điểm booking |
| Driver — thuộc tính | `vehicle_type` | Dùng làm filter hoặc feature |
| Booking | `trip_distance_km` (đón → trả), `passenger_type` | |
| Context | `hour_of_day` / `time_of_day`, `day_of_week`, `weather`, `traffic_level` | |
| Derived | `pickup_to_trip_ratio` (quãng đón / quãng chuyến); độ tin cậy của rating (rating 5.0 với 3 chuyến kém tin cậy hơn 4.8 với 2.000 chuyến); **relative features trong booking** (hạng khoảng cách, `distance − min_distance` của booking) | Relative features giúp pointwise model tập trung vào so sánh **trong** booking |

### 5.3 Data leakage thường gặp

| Loại leakage | Ví dụ | Cách tránh |
|---|---|---|
| Target leakage | Dùng `cancelled` làm feature để dự đoán `accepted` (chỉ có sau khi đã nhận) | Chỉ dùng thông tin có tại thời điểm offer |
| Temporal leakage | `acceptance_rate` tính trên toàn bộ dataset, gồm cả giai đoạn test | Tính "as-of" trước booking, hoặc dùng profile tĩnh trước cửa sổ mô phỏng |
| Group leakage | Chia ngẫu nhiên theo dòng: candidate của cùng một booking nằm ở cả train và test | Chia theo booking |
| Temporal split sai | Chia ngẫu nhiên khi dữ liệu có xu hướng theo thời gian | Train (sớm) → validation → test (muộn nhất) |
| Preprocessing leakage | Fit encoder / scaler / target encoding trên toàn bộ dữ liệu | Chỉ fit trên train |
| Oracle leakage (riêng synthetic) | Dùng xác suất ẩn của simulator làm feature | Lưu tách riêng; chỉ code evaluation được đọc |

**Khuyến nghị:** split **theo thời gian, ở cấp booking**.

---

## 6. Quyết định

### 6.1 Baseline: Nearest Driver

Định nghĩa vận hành (chi tiết chốt ở Phase 4):

1. Candidate generation **giống hệt** ML: available, trong bán kính $R$, loại xe phù hợp.
2. Sắp theo haversine distance tăng dần; hoà thì theo `driver_id` để kết quả deterministic.
3. Offer tuần tự tối đa $N$; outcome lấy từ simulator (ADR-005).

**Lý do.** Là mặc định trực quan và phổ biến, không cần train. Tối ưu trực tiếp guardrail ETA → là đối thủ khó ở Average ETA.
ML phải thắng MSR mà không vượt ngưỡng ETA (PRD H1–H2).

**Rủi ro "baseline quá yếu".** Nearest Driver bỏ qua hoàn toàn hành vi tài xế, nên ML có thể thắng dễ dàng. Đề xuất thêm baseline phụ
*weighted rule score* (PRD Q9) để tách phần cải thiện đến từ **việc học** khỏi phần đến từ **việc dùng thêm feature**.

### 6.2 ML approach: pointwise P(accept) với XGBoost

| Thành phần | Quyết định |
|---|---|
| Target | `accepted` (từng offer) |
| Model chính | `XGBClassifier`, objective `binary:logistic` |
| Mốc tuyến tính | `LogisticRegression` — đo giá trị của quan hệ phi tuyến |
| Fallback | `HistGradientBoostingClassifier` nếu XGBoost gây vấn đề dependency/deployment |
| Ranking | Sort $\hat{p}$ giảm dần trong candidate set → Top-K |
| Explanation | TreeSHAP qua `pred_contribs=True` (có sẵn trong XGBoost) → reason template (ADR-007) |
| Model selection | ROC-AUC / log loss và NDCG@5 trên validation; test set chỉ dùng **một lần** |
| Tuning | Random search nhỏ + early stopping trên validation |

**Lý do chọn:**

1. **Đúng bài toán** — sort theo xác suất là tối ưu dưới giả định của MVP (§2.3).
2. **Khớp dạng dữ liệu** — nhãn theo từng offer; không đòi hỏi nhãn cho mọi candidate.
3. **Xác suất dùng lại được** — utility kết hợp ETA (PRD Q2), trọng số cho optimization (future work), analytics.
4. **Gradient boosting là lựa chọn mặc định mạnh cho tabular cỡ vừa** [Grinsztajn et al., 2022]: bắt phi tuyến và tương tác, không cần scale feature, xử lý missing value.
5. **Giải thích được** — XGBoost tính TreeSHAP chính xác có sẵn [Lundberg et al., 2020], không cần thêm thư viện.
6. **Vận hành đơn giản** — inference trên CPU cho vài chục candidate, artifact nhỏ (sẽ đo ở Phase 7, 17).

**Vì sao không chọn các phương án khác cho MVP:**

| Phương án | Lý do |
|---|---|
| Learning-to-Rank | Cần nhãn cho nhiều candidate mỗi booking; score không calibrate; giữ làm ablation |
| Deep learning | Không có lợi thế rõ trên dữ liệu tabular cỡ này; tuning và hạ tầng nặng hơn; khó giải thích |
| Optimization-based | Cần simulator động với nhiều booking đồng thời; không khớp use case Top-K; future work |
| Reinforcement learning | Vượt phạm vi dữ liệu và mục tiêu học tập của MVP |
| LightGBM thay XGBoost | Năng lực tương đương; không có lý do đủ mạnh để đổi |

**Rủi ro của hướng đã chọn:**

| Rủi ro | Mitigation |
|---|---|
| ROC-AUC cao nhưng xếp hạng trong booking kém (model học hiệu ứng cấp booking) | Báo cáo metric theo nhóm (NDCG@5, Hit@1); thêm relative features |
| Selection bias từ logging policy | Dữ liệu training sinh bằng logging policy có exploration (PRD Q8, Phase 2) |
| $\hat{p}$ không calibrate khi dùng cho utility | Kiểm tra reliability curve / Brier score; calibrate trên validation nếu cần |
| Class imbalance | Dùng PR-AUC. Tránh reweight class (`scale_pos_weight`) nếu cần xác suất calibrate — ranking vốn không phụ thuộc threshold |

---

## 7. Experiment design

### 7.1 Ba tầng đánh giá

| Tầng | Câu hỏi | Metrics | Dữ liệu |
|---|---|---|---|
| ML | Model phân biệt nhận / từ chối tốt không? | ROC-AUC, PR-AUC, log loss, Brier, Precision/Recall/F1 | Các cặp có nhãn trong test |
| Ranking | Thứ tự trong từng booking có đúng không? | NDCG@5, Hit@1 | Candidate set của booking test |
| Business | Policy có tốt hơn khi vận hành không? | MSR, AR, Average ETA, CR | Simulator chạy trên booking test |

### 7.2 Offline policy evaluation

Log thực tế chỉ có outcome cho tài xế **đã được** offer, nên không trả lời được "nếu offer cho người khác thì sao" (counterfactual).
Ngoài đời, các cách xử lý gồm replay [Li et al., 2011], inverse propensity scoring / doubly robust [Dudík et al., 2011], và chuẩn vàng là A/B test online.

SmartMatch dùng **simulator làm oracle** (ADR-005), kèm kỹ thuật **common random numbers**:
mỗi cặp (booking, driver) được gán sẵn một số ngẫu nhiên cố định $u_{b,d} \sim U(0, 1)$; tài xế nhận chuyến khi và chỉ khi $u_{b,d} < p_{b,d}$.
Mọi policy thấy **cùng một "thế giới"** → chênh lệch metric chỉ đến từ thứ tự offer, và phương sai của phép so sánh giảm mạnh.

### 7.3 Các policy được so sánh

| Policy | Vai trò | Priority |
|---|---|---|
| Random (trong candidate set) | Sàn — sanity check | C |
| Nearest Driver | Baseline chính | M |
| Weighted rule score | Baseline phụ | S (Q9) |
| ML (XGBoost) | Đề xuất | M |
| Oracle (sort theo xác suất ẩn) | Trần lý thuyết, không đạt được ngoài đời | S (Q9) |

Khi có Oracle, báo cáo thêm **gap closed** = (ML − Baseline) / (Oracle − Baseline): ML thu được bao nhiêu phần của mức cải thiện tối đa có thể.

### 7.4 Độ tin cậy thống kê

Dùng **paired bootstrap theo booking** [Efron & Tibshirani, 1993]: lấy mẫu lại tập booking test (ví dụ 1.000 lần), tính chênh lệch metric
giữa hai policy trên cùng mẫu, báo cáo khoảng tin cậy 95%. Chỉ kết luận "cải thiện" khi khoảng tin cậy không chứa 0.

### 7.5 Ablation plan

Mỗi ablation chỉ thay đổi **một** yếu tố; cùng split, cùng seed.

| ID | Thí nghiệm | Câu hỏi | Priority |
|---|---|---|---|
| AB1 | Logistic Regression vs XGBoost | Quan hệ phi tuyến có đáng giá không? | M |
| AB2 | Bỏ nhóm feature lịch sử tài xế | Lịch sử hành vi đóng góp bao nhiêu? | M |
| AB3 | Bỏ nhóm context | Traffic / weather / time có ý nghĩa không? | S |
| AB4 | Chỉ dùng distance / ETA | ML với ít feature có hơn Nearest Driver không? | S |
| AB5 | Bỏ relative features | Relative features có giúp xếp hạng trong booking không? | S |
| AB6 | `XGBClassifier` vs `XGBRanker` | LTR có tốt hơn khi đủ nhãn không? | C (phụ thuộc Q8) |
| AB7 | Score = $\hat{p}$ vs utility có ETA / cancellation | Trade-off MSR – ETA – CR (PRD Q2) | S |

### 7.6 Latency benchmarking (preview Phase 17, 20)

Đo inference cho một booking với 10 / 20 / 50 candidates, báo cáo P50 / P95 / P99, tách riêng: build feature, predict, tính SHAP
(`pred_contribs` tốn thêm thời gian so với predict thường).

---

## 8. Yêu cầu đầu vào cho Phase 2 (synthetic data)

1. **Mô hình hành vi ẩn** dạng `p = sigmoid(f(features) + tương tác phi tuyến + hiệu ứng ẩn theo tài xế + nhiễu)`:
   có phi tuyến để so sánh Logistic Regression vs XGBoost có ý nghĩa; có yếu tố không quan sát được để AUC không phi thực tế (PRD H4).
2. Mỗi booking có một **candidate set** gồm các tài xế trong bán kính, lưu đủ feature.
3. **Logging policy** sinh nhãn training có exploration; quyết định nhãn chỉ cho tài xế được offer hay cho mọi candidate (PRD Q8).
4. Xác suất ẩn và random numbers $u_{b,d}$ **lưu tách riêng** (oracle), chỉ code evaluation được đọc.
5. `request_time` trải trên nhiều tuần để split theo thời gian.
6. Feature lịch sử tài xế là giá trị **trước** cửa sổ mô phỏng (hoặc tính as-of).
7. Mô hình **huỷ chuyến** ẩn riêng để tính Cancellation Rate.
8. Không PII; seed cố định; kích thước cấu hình được.

## 9. EDA findings (Phase 3)

Nguồn: [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb). Phạm vi: **chỉ tập train** theo `ml/data/splits.py`
(2026-06-01 → 2026-07-10: 71,863 booking = 71.9%; 558,334 candidate; 108,050 logged offers). Không đọc `data/oracle/`.

> **Lưu ý về tính độc lập.** Simulator và danh sách feature ứng viên do cùng một người thiết kế, nên việc EDA "tìm lại" được cấu trúc
> của mô hình sinh dữ liệu **không phải bằng chứng độc lập**. Giá trị của phần này là luyện quy trình EDA và nhận diện các bẫy
> (confounding, selection bias, leakage) sẽ gặp trong dữ liệu thật — nơi không ai biết mô hình sinh dữ liệu.

### 9.1 Chất lượng dữ liệu

| Kiểm tra | Kết quả |
|---|---|
| Trùng khoá (drivers, bookings, candidates) | 0 |
| Structural missing | `offer_rank`, `accepted` trống ở 80.65% dòng candidate (= không được offer); `cancelled` trống ở 89.59% (= không được offer hoặc không nhận). Khớp 100% với logic của log |
| Missing thật | `rating`: 292 tài xế (1.46%), tất cả có 0–4 chuyến → missing có thông tin |
| Hai cột idle time khác nghĩa | `drivers.idle_time_min` là snapshot hiện tại (cho demo); `candidates.idle_time_min` là giá trị tại thời điểm booking → training phải dùng cột của `candidates` |

### 9.2 Nhãn & cấu trúc log

| | |
|---|---|
| Acceptance rate (logged offers) | 0.538 → cân bằng |
| Cancellation rate (chuyến đã nhận) | 0.133 → lệch; chỉ liên quan nếu mô hình hoá việc huỷ (PRD Q2) |
| Tỷ lệ dòng candidate có nhãn | 19.4% |
| Acceptance theo lượt offer 1 / 2 / 3 — policy `nearest` | 0.702 / 0.398 / 0.214 |
| Acceptance theo lượt offer 1 / 2 / 3 — policy `explore` | 0.415 / 0.355 / 0.313 |
| Acceptance theo ngày | mean 0.540, std 0.046, min 0.435, max 0.613; không thấy xu hướng trong 40 ngày |

### 9.3 Phân phối & outlier

- Tài xế **được offer** có ETA và distance ngắn hơn hẳn toàn bộ candidate (offered: median ETA 10.3 phút, distance 2.26 km) —
  selection bias của log nhìn thấy trực tiếp trên histogram.
- Mật độ candidate theo distance tăng tuyến tính tới 5 km: hệ quả hình học của phân bố đều trong hình tròn (vòng ngoài có diện tích lớn hơn).
- `rating` lệch trái (p5 4.35, p50 4.66) và dồn tại 5.0 do bị chặn; `cancellation_rate` lệch phải với nhiều giá trị 0;
  `completed_trips` gần log-normal kèm một cụm tài xế mới.
- Outlier theo IQR: `idle_time_min` 6.0% vượt fence 27.9 phút (max 180); `completed_trips` 8.2% vượt 1,687; `cancellation_rate` 4.6%
  vượt 0.23 (max 0.70); ETA 0.95% vượt 30.7 phút; `trip_km` 0.25%; `rating` 0.73% dưới 4.20.
  Đây là giá trị hợp lệ của phân phối lệch, không phải lỗi → **giữ lại**. Tree model không nhạy; Logistic Regression dùng `log1p`.

### 9.4 Quan hệ một biến với việc nhận chuyến

| Feature | Quan hệ quan sát được (acceptance rate theo bin) |
|---|---|
| `estimated_eta_min` | Giảm mạnh, dạng chữ S: ≈ 0.92 ở 1–2 phút, ≈ 0.78 ở 7–8 phút, dốc nhất khoảng 8–16 phút, ≈ 0.01 ở 30 phút |
| `distance_km` | Giảm (≈ 0.88 → 0.14) |
| `idle_time_min` | Tăng rồi bão hoà (≈ 0.47 → 0.69) |
| `trip_km` | Tăng, dạng lõm (≈ 0.38 ở ~1.6 km → 0.61 ở ~17 km) |
| `acceptance_rate` (lịch sử) | Tăng đơn điệu (≈ 0.28 → 0.76) |
| `cancellation_rate`, `rating`, `completed_trips` | Gần như phẳng quanh mức chung |
| `weather` | clear 0.588 · rain 0.468 · heavy_rain 0.233 |
| `traffic_level` | low 0.802 · medium 0.676 · high 0.336 |
| `time_of_day` | night 0.716 · morning_peak 0.459 · midday 0.666 · evening_peak 0.417 · evening 0.689 |
| `passenger_type` | individual 0.541 · group 0.482 |
| `vehicle_type`, `is_weekend` | Không khác biệt đáng kể (car_4 0.539 vs car_7 0.534; weekday 0.536 vs weekend 0.545) |

### 9.5 Confounding & Simpson's paradox

So sánh acceptance rate **trong cùng khoảng ETA** để tách tác động của biến ngữ cảnh khỏi tác động của ETA:

| Biến | Chênh lệch chung | Trong cùng khoảng ETA | Kết luận |
|---|---|---|---|
| `traffic_level` | low 0.802 vs high 0.336 (median ETA 5.7 vs 15.2 phút) | low ≈ medium (0.892 vs 0.873 ở 0–6 phút; 0.774 vs 0.773 ở 6–10); high thấp hơn medium ≈ 0.07–0.12 | Phần lớn chênh lệch đi qua ETA; traffic cao vẫn còn tác động riêng |
| `time_of_day` | peak 0.417–0.459 vs midday 0.666 (median ETA 12.4–13.4 vs 7.7 phút) | Chênh lệch giữa các khung giờ thu hẹp còn ≈ 0.05–0.10 | Chủ yếu do ETA |
| `weather` | heavy_rain 0.233 vs clear 0.588 | heavy_rain thấp hơn clear 0.15 (0–6 phút), 0.21 (6–10), 0.27 (10–15) | Tác động riêng, **lớn dần theo ETA** → dấu hiệu tương tác weather × ETA |
| `passenger_type` | group 0.482 < individual 0.541 | group **cao hơn** ở mọi khoảng ETA (0.805 vs 0.742 ở 6–10 phút; 0.550 vs 0.474 ở 10–15) | **Simpson's paradox:** group chỉ có median 2 candidate phù hợp (vs 7) → tài xế xa hơn (median ETA 13.2 vs 10.1 phút) |

### 9.6 Selection bias

| Nhóm offer đầu tiên | Offers | Acceptance | Median ETA | Median candidate / booking |
|---|---|---|---|---|
| Policy `nearest` | 56,428 | 0.702 | 6.96 phút | 7 |
| Policy `explore` (ngẫu nhiên) | 14,371 | 0.415 | 14.20 phút | 7 |
| `explore`, trúng tài xế gần nhất | 2,780 | 0.606 | 8.65 phút | 4 |

- Cùng là "tài xế gần nhất" nhưng acceptance 0.606 vs 0.702: chọn ngẫu nhiên trúng người gần nhất dễ xảy ra hơn ở booking có ít candidate,
  nơi người gần nhất vẫn ở xa hơn. → **Một điều kiện lọc tưởng vô hại cũng tạo ra bias.**
- Offer đầu tiên ngẫu nhiên theo hạng khoảng cách trong booking: 0.606 (gần nhất) → 0.492 → 0.426 → 0.359 → ≈ 0.31–0.32 (hạng 5–7)
  → 0.282 (hạng ≥ 8). Khoảng cách là tín hiệu chính nhưng không tuyệt đối: tài xế gần thứ hai vẫn nhận gần một nửa số lần.
  Đây là **dấu hiệu có không gian** để xếp hạng tốt hơn Nearest Driver — **chưa phải bằng chứng** ML sẽ thắng.
- `offer_rank` và `logging_policy` gắn chặt với nhãn vì là sản phẩm của log (lượt sau chỉ xảy ra khi lượt trước bị từ chối)
  → **không được làm feature**.

### 9.7 Tương quan & feature importance sơ bộ

Spearman trên logged offers:

- ETA ↔ distance 0.90; `distance_minus_min_km` ↔ `distance_rank_in_booking` 0.97 (gần như trùng nhau);
  `pickup_to_trip_ratio` ↔ distance 0.72 và ↔ `trip_km` −0.64.
- Feature lịch sử của tài xế gần như không tương quan với feature không gian → tín hiệu bổ sung. `rating` ↔ `cancellation_rate` −0.34.
- Với `accepted`: ETA −0.585 · distance −0.494 · `pickup_to_trip_ratio` −0.432 · `distance_minus_min_km` −0.376 ·
  `distance_rank_in_booking` −0.369 · `acceptance_rate` +0.245 · `idle_time_min` +0.119 · `trip_km` +0.108; các feature còn lại có |ρ| ≤ 0.03.

Importance một biến (60,000 logged offers lấy mẫu từ tập train):

| Feature | Mutual information | Univariate ROC-AUC | Chiều |
|---|---|---|---|
| `estimated_eta_min` | 0.201 | 0.837 | cao hơn → ít nhận hơn |
| `distance_km` | 0.132 | 0.785 | cao hơn → ít nhận hơn |
| `pickup_to_trip_ratio` | 0.102 | 0.749 | cao hơn → ít nhận hơn |
| `distance_rank_in_booking` | 0.071 | 0.690 | cao hơn → ít nhận hơn |
| `distance_minus_min_km` | 0.070 | 0.694 | cao hơn → ít nhận hơn |
| `traffic_level` | 0.070 | 0.693 | cao hơn → ít nhận hơn |
| `trip_km` | 0.046 | 0.560 | cao hơn → nhận nhiều hơn |
| `acceptance_rate` | 0.032 | 0.642 | cao hơn → nhận nhiều hơn |
| `hour` | 0.030 | 0.505 | không đơn điệu |
| `time_of_day` | 0.030 | — (nominal) | — |
| `weather` | 0.017 | 0.577 | xấu hơn → ít nhận hơn |
| `idle_time_min` | 0.008 | 0.569 | cao hơn → nhận nhiều hơn |
| `cancellation_rate`, `completed_trips`, `rating` | ≤ 0.003 | 0.49–0.51 | ≈ 0 |
| `passenger_type`, `has_rating`, `is_weekend`, `vehicle_type` | < 0.001 | ≈ 0.50 | ≈ 0 |

Hai thước đo không luôn cùng thứ tự: `hour` có AUC ≈ 0.5 nhưng mutual information 0.030 (quan hệ không đơn điệu — hai khung cao điểm);
`passenger_type` có mutual information ≈ 0 dù có tác động rõ khi so trong cùng khoảng ETA (§9.5). Importance một biến chỉ là điểm xuất phát.

### 9.8 Hệ quả cho Phase 4–5

| # | Quyết định / việc cần làm | Căn cứ |
|---|---|---|
| 1 | Nearest Driver sẽ là baseline mạnh: khoảng cách / ETA là tín hiệu một biến mạnh nhất | §9.4, §9.7 |
| 2 | ML phải khai thác phần tín hiệu khoảng cách không có: lịch sử tài xế, idle time, độ dài chuyến, thời tiết / traffic và tương tác với ETA | §9.4, §9.5 |
| 3 | Feature ứng viên cho Phase 5: `estimated_eta_min`, `distance_km`, `pickup_to_trip_ratio`, `trip_km`, `acceptance_rate`, `idle_time_min`, `traffic_level`, `weather`, `time_of_day`, và **một** trong hai relative feature (ρ = 0.97) | §9.7 |
| 4 | Không loại `rating`, `cancellation_rate`, `completed_trips`, `passenger_type` chỉ vì importance một biến thấp — để ablation quyết định | §9.5 (`passenger_type`) |
| 5 | Cấm làm feature: `offer_rank`, `logging_policy`, `cancelled`, `drivers.idle_time_min` (snapshot hiện tại), mọi thứ trong `data/oracle/` | §9.1, §9.6 |
| 6 | Preprocessing: XGBoost dùng NaN native cho `rating`; Logistic Regression cần impute + `has_rating`, one-hot cho categorical, `log1p` cho `completed_trips` và `idle_time_min`, chuẩn hoá | §9.1, §9.3 |
| 7 | Không cần resampling hay class weight cho `accepted` | §9.2 |
| 8 | Báo cáo metric theo booking (NDCG@5, Hit@1) bên cạnh ROC-AUC: traffic, weather, time là tín hiệu chung của cả booking — giúp AUC toàn cục nhưng không giúp xếp hạng trong booking | §9.4, §6.2 |
| 9 | Vùng tài xế xa chủ yếu có nhãn từ 20% explore → model kém chắc chắn hơn ở vùng này; kiểm tra calibration theo khoảng ETA ở Phase 6 | §9.6 |

---

## References

Các tài liệu có link đã được kiểm tra tồn tại trực tuyến trong Phase 1 (2026-09-15).
Các tài liệu kinh điển không có link được trích từ hiểu biết chung — cần kiểm tra lại nếu dùng cho trích dẫn chính thức.

1. Alonso-Mora, J., Samaranayake, S., Wallar, A., Frazzoli, E., Rus, D. (2017). On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment. *PNAS*, 114(3), 462–467. https://doi.org/10.1073/pnas.1611675114
2. Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. *Microsoft Research Technical Report MSR-TR-2010-82*. https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
3. Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., Li, H. (2007). Learning to Rank: From Pairwise Approach to Listwise Approach. *ICML 2007*.
4. Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*. https://arxiv.org/abs/1603.02754
5. Covington, P., Adams, J., Sargin, E. (2016). Deep Neural Networks for YouTube Recommendations. *RecSys 2016*.
6. Dudík, M., Langford, J., Li, L. (2011). Doubly Robust Policy Evaluation and Learning. *ICML 2011*. https://arxiv.org/abs/1103.4601
7. Efron, B., Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
8. Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022*. https://arxiv.org/abs/2207.08815
9. Järvelin, K., Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4).
10. Joachims, T. (2002). Optimizing Search Engines using Clickthrough Data. *KDD 2002*.
11. Joachims, T., Swaminathan, A., Schnabel, T. (2017). Unbiased Learning-to-Rank with Biased Feedback. *WSDM 2017*. https://arxiv.org/abs/1608.04468
12. Karp, R. M., Vazirani, U. V., Vazirani, V. V. (1990). An optimal algorithm for on-line bipartite matching. *STOC 1990*.
13. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
14. Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2.
15. Li, L., Chu, W., Langford, J., Wang, X. (2011). Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms. *WSDM 2011*. https://arxiv.org/abs/1003.5956
16. Liu, T.-Y. (2009). Learning to Rank for Information Retrieval. *Foundations and Trends in Information Retrieval*, 3(3).
17. Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NIPS 2017*. https://arxiv.org/abs/1705.07874
18. Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2. https://doi.org/10.1038/s42256-019-0138-9
19. Niculescu-Mizil, A., Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *ICML 2005*.
20. Qin, Z., Tang, X., Jiao, Y., Zhang, F., Xu, Z., Zhu, H., Ye, J. (2020). Ride-Hailing Order Dispatching at DiDi via Reinforcement Learning. *INFORMS Journal on Applied Analytics*, 50(5), 272–286. https://doi.org/10.1287/inte.2020.1047
21. Wang, H., Yang, H. (2019). Ridesourcing systems: A framework and review. *Transportation Research Part B: Methodological*, 129, 122–155.
22. Xu, Z., Li, Z., Guan, Q., Zhang, D., Li, Q., Nan, J., Liu, C., Bian, W., Ye, J. (2018). Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach. *KDD 2018*. https://doi.org/10.1145/3219819.3219824
23. Yan, C., Zhu, H., Korolko, N., Woodard, D. (2020). Dynamic pricing and matching in ride-hailing platforms. *Naval Research Logistics*, 67(8), 705–724. https://doi.org/10.1002/nav.21872
24. Zhang, L., Hu, T., Min, Y., Wu, G., Zhang, J., Feng, P., Gong, P., Ye, J. (2017). A Taxi Order Dispatch Model based On Combinatorial Optimization. *KDD 2017*. https://doi.org/10.1145/3097983.3098138

**Tools & data**

- XGBoost documentation — Learning to Rank: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
- XGBoost documentation — Prediction (`pred_contribs`, TreeSHAP): https://xgboost.readthedocs.io/en/stable/prediction.html
- NYC Taxi & Limousine Commission — TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
