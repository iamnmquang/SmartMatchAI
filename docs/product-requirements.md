# SmartMatch AI — Product Requirements Document (PRD)

| | |
|---|---|
| Version | 0.4 |
| Phase | 0 — Product Definition; §8.2 và Q2 cập nhật ở Phase 5 |
| Status | Draft — chờ review |
| Owner | @iamnmquang |
| Last updated | 2026-09-16 |

> **Disclaimer.** SmartMatch AI là project **mô phỏng** phục vụ học tập. Toàn bộ dữ liệu là synthetic.
> Hệ thống không dựa trên, không đại diện và không giả định kiến trúc hay logic production của bất kỳ công ty nào.
> Mọi con số "target" trong tài liệu này là **giả thuyết cần kiểm chứng**, không phải kết quả.

---

## 1. Problem statement

### 1.1 Bối cảnh

Trên một nền tảng gọi xe, mỗi booking mới cần được gán cho một tài xế. Quyết định này ảnh hưởng tới cả ba bên:

- **Hành khách**: chờ lâu (ETA cao), hoặc booking không có ai nhận, hoặc bị huỷ sau khi đã có tài xế.
- **Tài xế**: nhận được chuyến không phù hợp → từ chối hoặc huỷ.
- **Nền tảng**: booking không được ghép thành công là doanh thu bị mất và trải nghiệm xấu.

Cách đơn giản nhất là **gán tài xế gần nhất**. Nhưng tài xế gần nhất chưa chắc sẽ nhận chuyến: họ có thể có acceptance rate thấp,
hay huỷ chuyến, hoặc điều kiện traffic/thời tiết/khung giờ khiến chuyến kém hấp dẫn. Khi tài xế đầu tiên từ chối, hệ thống phải
gửi offer cho người tiếp theo → hành khách chờ thêm.

Các yếu tố cần cân nhắc khi chọn tài xế:

| Nhóm | Yếu tố |
|---|---|
| Không gian / thời gian | Distance, ETA |
| Chất lượng & hành vi tài xế | Rating, acceptance rate, cancellation rate, completed trips, idle time |
| Ngữ cảnh | Traffic, time of day, weather |

### 1.2 Problem statement

> Operations team cần một cách **xếp hạng tài xế ứng viên** cho mỗi booking sao cho
> **tăng khả năng booking được nhận thành công**, **không làm ETA xấu đi đáng kể**,
> và **giải thích được** vì sao một tài xế được ưu tiên.

### 1.3 Tại sao cân nhắc Machine Learning?

Quan hệ giữa các yếu tố trên và hành vi nhận chuyến là **phi tuyến và có tương tác** (ví dụ: idle time dài làm tăng khả năng nhận,
nhưng mưa + giờ cao điểm + quãng đường đón xa lại làm giảm). Viết rule tay cho mọi tổ hợp thì khó bảo trì và khó tối ưu.

Tuy nhiên, **ML chỉ đáng dùng nếu thắng baseline** (Nearest Driver) trên business metrics mà không vi phạm guardrail.
Đó là giả thuyết trung tâm của project — và kết luận "ML không đáng dùng" cũng là một kết quả hợp lệ.

---

## 2. Goals & Non-goals

### 2.1 Goals

| ID | Goal |
|---|---|
| G1 | Xếp hạng Top-K tài xế cho một booking bằng ML model, kèm lý do. |
| G2 | Đo được hiệu quả so với baseline Nearest Driver bằng business metrics và ML metrics, trên cùng điều kiện. |
| G3 | Cho phép Ops/Analyst/PM hỏi đáp bằng ngôn ngữ tự nhiên về matching, số liệu và policy — câu trả lời dựa trên dữ liệu thật của hệ thống (tool) hoặc tài liệu (RAG), không bịa. |
| G4 | Chạy end-to-end bằng `docker compose up` và deploy được lên VPS với HTTPS. |

### 2.2 Non-goals (không xây dựng)

- Mobile app cho tài xế hoặc hành khách.
- Thanh toán, pricing, surge pricing.
- GPS tracking real-time thật, routing engine thật (ETA chỉ là ước lượng).
- Đặt xe thật, giao tiếp với tài xế thật.
- Tối ưu toàn cục cho nhiều booking cùng lúc (global assignment) — chỉ nghiên cứu ở Phase 1, không nằm trong MVP.
- Kubernetes, microservices.

---

## 3. Users & Personas

### P1 — Operations / Admin (Primary)

| | |
|---|---|
| Công việc | Giám sát việc ghép chuyến, xử lý booking khó ghép, kiểm tra vì sao hệ thống chọn một tài xế. |
| Pain points | Không biết tại sao booking bị từ chối nhiều; phải tự tra nhiều bảng dữ liệu. |
| Cần | Chạy matching cho một booking và xem Top-K kèm lý do; dashboard KPI; hỏi nhanh "booking 123 nên giao cho ai?". |

### P2 — Analyst (Secondary)

| | |
|---|---|
| Công việc | Theo dõi KPI, tìm nguyên nhân khi metric thay đổi. |
| Pain points | Viết đi viết lại query cho các câu hỏi ad-hoc. |
| Cần | KPI theo thời gian / khung giờ / thời tiết / traffic; hỏi "tại sao acceptance rate giảm?" và nhận phân tích dựa trên số liệu thật. |

### P3 — Product Manager (Secondary)

| | |
|---|---|
| Công việc | Quyết định có dùng model mới hay không; hiểu trade-off. |
| Pain points | Khó so sánh model mới với cách làm hiện tại một cách công bằng. |
| Cần | Bảng Baseline vs ML; thông tin model version; policy matching dạng tài liệu tra cứu được. |

---

## 4. User journeys

### J1 — Ops chạy matching cho một booking

1. Mở **Matching Simulator** → chọn booking.
2. Bấm **Run matching**.
3. Hệ thống: lấy tài xế ứng viên trong bán kính → tính feature → model chấm điểm → trả Top-5.
4. Ops xem bảng *Rank / Driver / Score / ETA / Rating / Acceptance rate* kèm lý do.
5. Kết quả được lưu, xem lại qua `GET /matching/{booking_id}`.

Edge cases: booking không tồn tại → 404 rõ ràng; không có ứng viên trong bán kính → thông báo rõ, không trả lỗi 500.

### J2 — Analyst điều tra acceptance rate

1. Mở **Dashboard** → thấy acceptance rate thấp hơn kỳ trước.
2. Hỏi **AI Assistant**: "Tại sao acceptance rate giảm?"
3. Agent: kiểm tra scope → gọi analytics tool (breakdown theo khung giờ / thời tiết / traffic) → LLM phân tích **chỉ trên số liệu tool trả về**.
4. UI hiển thị câu trả lời + tool đã gọi + số liệu gốc.

### J3 — PM đánh giá model và policy

1. Xem thông tin model: version, ngày train, metric trên test set.
2. Đọc evaluation report Baseline vs ML.
3. Hỏi assistant: "Quy định matching là gì?" → RAG trả lời kèm trích dẫn tài liệu nguồn.

---

## 5. MVP scope

| # | MVP item | Mô tả | Phase |
|---|---|---|---|
| 1 | Generate dataset | Synthetic drivers, bookings, context, matching observations; tái lập được bằng seed | 2–3 |
| 2 | Train model | Baseline Nearest Driver + ML model dự đoán khả năng nhận chuyến | 4–6 |
| 3 | Ranking drivers | Inference: candidates → score → Top-K + reason | 7 |
| 4 | API | FastAPI + PostgreSQL | 8–9 |
| 5 | Dashboard | React: KPI + Matching Simulator + Chat UI | 14 |
| 6 | AI Assistant | LLM + tools + RAG + LangGraph agent + guardrails | 10–13 |

Nền tảng vận hành bắt buộc cho một "mini production product" (không phải feature cho user): Docker, testing, monitoring,
deployment, CI/CD (Phase 15–19).

---

## 6. Functional requirements

Priority: **M** = Must (MVP) · **S** = Should · **C** = Could.

### 6.1 Data & ML

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Sinh dataset synthetic (drivers, bookings, context, matching observations) với random seed cố định; kích thước cấu hình được (mục tiêu 100k bookings, 20k drivers). | M |
| FR-02 | Ground truth sinh từ một mô hình xác suất **có nhiễu**, được document; dữ liệu không được dễ tới mức model đạt kết quả phi thực tế. | M |
| FR-03 | Baseline Nearest Driver được đánh giá trên **cùng tập booking test** với ML. | M |
| FR-04 | Train model dự đoán `accepted` với train/validation/test split không data leakage. | M |
| FR-05 | Lưu model artifact kèm metadata: version, feature list, metric, thời điểm train. | M |
| FR-06 | Báo cáo Baseline vs ML chỉ dùng số liệu từ experiment thực tế đã chạy. | M |

### 6.2 Matching

| ID | Requirement | Priority |
|---|---|---|
| FR-07 | Candidate generation: lọc tài xế available trong bán kính cấu hình được. | M |
| FR-08 | Chấm điểm từng candidate bằng model, sort giảm dần, trả Top-K (mặc định K=5, có giới hạn tối đa). | M |
| FR-09 | Mỗi recommendation có `reason` nêu các yếu tố ảnh hưởng chính tới điểm. | M |
| FR-10 | Lưu kết quả matching cùng model version đã dùng. | M |
| FR-11 | Booking không tồn tại / không có candidate → lỗi có kiểm soát (404/422 hoặc response rỗng có giải thích), không 500. | M |

### 6.3 API

| ID | Requirement | Priority |
|---|---|---|
| FR-12 | Endpoints: `GET /health`, `GET /drivers`, `GET /bookings`, `POST /matching`, `GET /matching/{booking_id}`, `GET /analytics`, `GET /model/info`, `POST /chat`. | M |
| FR-13 | Endpoint danh sách có pagination. | S |
| FR-14 | OpenAPI documentation tự sinh. | M |
| FR-15 | Authentication cơ bản cho mọi endpoint ngoài `/health` khi deploy public. | M |

### 6.4 Analytics & Dashboard

| ID | Requirement | Priority |
|---|---|---|
| FR-16 | KPI: total bookings, matching success rate, acceptance rate, average ETA, cancellation rate. | M |
| FR-17 | Breakdown KPI theo ngày, khung giờ, thời tiết, traffic. | S |
| FR-18 | Matching Simulator: chọn booking → run → hiển thị Top-5. | M |
| FR-19 | Hiển thị so sánh Baseline vs ML trên UI. | C |

### 6.5 AI Assistant

| ID | Requirement | Priority |
|---|---|---|
| FR-20 | Trả lời câu hỏi về matching, drivers, bookings, system analytics, project policies. | M |
| FR-21 | Câu hỏi cần dữ liệu hệ thống **bắt buộc** đi qua tool (`get_driver`, `get_booking`, `run_matching`, `get_analytics`, `search_knowledge`); LLM không tự tạo số liệu. | M |
| FR-22 | Câu hỏi về policy trả lời dựa trên RAG, có citation tới tài liệu nguồn. | M |
| FR-23 | Câu hỏi ngoài scope → thông điệp chuẩn: *"Tôi chỉ hỗ trợ các vấn đề liên quan đến SmartMatch."* | M |
| FR-24 | Response kèm metadata: tool đã gọi, tài liệu nguồn. | M |
| FR-25 | Chạy được với `MockLLMProvider` khi không có API key (dev, test, CI). | M |
| FR-26 | Hội thoại nhiều lượt (conversation memory). | C |

---

## 7. Non-functional requirements

| ID | Loại | Requirement | Kiểm chứng |
|---|---|---|---|
| NFR-01 | Performance | `POST /matching` P95 < 300 ms trên máy dev với ≤ 50 candidates (target, không gồm LLM). | Benchmark — Phase 17, 20 |
| NFR-02 | Performance | `POST /chat` P95 < 15 s (phụ thuộc LLM provider; target). | Benchmark — Phase 20 |
| NFR-03 | Reproducibility | Cùng code + seed + config → cùng dataset và cùng metric (nondeterminism của thư viện nếu có phải được ghi chú). | Chạy lại pipeline |
| NFR-04 | Explainability | `reason` được sinh **deterministic** từ feature/model, không phải do LLM tự nghĩ. | Test |
| NFR-05 | Security | Không hard-code secret; secret chỉ nằm trong `.env` (không commit). | Review + `git check-ignore` |
| NFR-06 | Security | LLM không bao giờ sinh hoặc chạy raw SQL; chỉ gọi hàm định nghĩa trước có validate tham số. | Agent tests |
| NFR-07 | Security | Nội dung tài liệu retrieve được không thể override system instruction. | Prompt-injection tests |
| NFR-08 | Privacy | Không sinh PII (tên, số điện thoại, biển số). API/assistant chỉ trả các trường cần thiết; không lộ config/secret. | Review schema |
| NFR-09 | Reliability | Lỗi hoặc timeout của LLM provider không làm sập API; trả lỗi có kiểm soát. | Tests |
| NFR-10 | Observability | Log có `request_id`, endpoint, latency, status; matching log `model_version`, prediction latency; LLM log model, latency, token usage. | Phase 17 |
| NFR-11 | Portability | `docker compose up` chạy toàn bộ stack, không phụ thuộc OS của máy dev. | Phase 15 |
| NFR-12 | Maintainability | Đổi LLM provider/model chỉ bằng config; core backend coverage ≥ 70% với test có ý nghĩa. | Phase 16 |
| NFR-13 | Cost | `/chat` có giới hạn token/request và rate limit khi deploy public. | Phase 13, 18 |

---

## 8. Metrics & KPIs

### 8.1 Business metrics

Để ranking có ý nghĩa vượt ra ngoài "chọn 1 người", project giả định **offer tuần tự**: gửi offer cho tài xế hạng 1; nếu từ chối thì
gửi cho hạng 2; tối đa `N` lượt (đề xuất `N = 3`).

| Metric | Định nghĩa (đề xuất) | Hướng tốt |
|---|---|---|
| **Matching Success Rate (MSR)** | % booking có tài xế chấp nhận trong tối đa `N` lượt offer. | ↑ |
| **Acceptance Rate (AR)** | Số offer được chấp nhận / tổng số offer đã gửi. | ↑ |
| **Average ETA** | ETA đón khách trung bình (phút) của tài xế **đã nhận chuyến**, tính trên các booking matched. | ↓ |
| **Cancellation Rate (CR)** | % booking matched bị huỷ sau khi tài xế đã nhận. | ↓ |
| **Completed Rate** | % booking được nhận **và** không bị huỷ, trên mọi booking. Kết hợp MSR và CR. | ↑ |

Lưu ý khi đọc metric:

- **Average ETA chỉ tính trên booking matched → có selection bias.** Một policy ghép được ít booking hơn có thể có ETA trung bình thấp hơn.
  Luôn đọc Average ETA cùng với MSR.
- AR phụ thuộc vào `N`. Định nghĩa đã được **chốt ở Phase 4** ([evaluation.md §2.3](evaluation.md)) và **phải dùng giống hệt nhau** cho Baseline và ML.

### 8.2 North Star & guardrails

- **North Star metrics: Matching Success Rate và Completed Rate** (cập nhật ở Phase 5) — booking được nhận là giá trị cốt lõi cho
  hành khách, tài xế và nền tảng; nhưng một chuyến được nhận rồi huỷ không tạo ra giá trị đó.
  Lý do nâng Completed Rate lên ngang hàng: kết quả Phase 4 cho thấy MSR gần như đã bão hoà (trần chỉ +1.2 pp so với Nearest Driver) và
  policy tối đa hoá riêng khả năng nhận (Oracle) **không** làm tăng số chuyến hoàn thành — xem [evaluation.md §5](evaluation.md).
  Khi hai metric mâu thuẫn, Completed Rate là metric quyết định.
- **Guardrail metrics: Average ETA, Cancellation Rate** — ML không được "thắng" North Star bằng cách đẩy ETA lên quá cao
  hoặc ưu tiên tài xế nhận rồi huỷ.

### 8.3 Success criteria — giả thuyết, kiểm chứng ở Phase 6

| ID | Hypothesis |
|---|---|
| H1 | ML ranking đạt MSR cao hơn Nearest Driver trên cùng tập booking test. |
| H1b | ML ranking đạt Completed Rate cao hơn Nearest Driver (bổ sung ở Phase 5 cùng với North Star thứ hai). |
| H2 | Average ETA của ML không tăng quá **+15%** so với baseline (ngưỡng đề xuất — là quyết định product, có thể điều chỉnh). |
| H3 | Cancellation Rate của ML không cao hơn baseline. |
| H4 | ROC-AUC trên test tốt hơn rõ rệt so với random (0.5) nhưng không phi thực tế. Nếu > 0.95 → nghi ngờ leakage hoặc dữ liệu quá dễ, phải điều tra trước khi báo cáo. |

Nếu H1 không đạt, evaluation report ghi đúng như vậy.

### 8.4 ML metrics

| Metric | Vai trò |
|---|---|
| ROC-AUC | Khả năng phân biệt accepted / not accepted, không phụ thuộc threshold. |
| PR-AUC | Bổ sung khi class imbalance. |
| Precision / Recall / F1 | Phụ thuộc threshold. **Ranking không cần threshold** (chỉ cần thứ tự điểm) nên các metric này mang tính chẩn đoán. |
| Ranking metrics (NDCG@K, Hit@1) | Đánh giá trực tiếp chất lượng thứ tự — cân nhắc ở Phase 6 (S). |

### 8.5 System & LLM metrics

| Nhóm | Metric |
|---|---|
| API | Latency P50/P95/P99 theo endpoint; error rate. |
| ML | Inference latency; model version đang phục vụ. |
| LLM | Latency, token usage (nếu provider hỗ trợ), tool-call success rate. |
| Assistant quality | Tỷ lệ chọn đúng tool và tỷ lệ từ chối đúng câu hỏi ngoài scope trên một bộ câu hỏi test cố định. |
| RAG | Retrieval hit rate trên bộ câu hỏi có nhãn nhỏ (C). |

---

## 9. Key trade-offs

| # | Trade-off | Cách xử lý trong MVP |
|---|---|---|
| T1 | **Acceptance vs ETA** — tài xế có khả năng nhận cao có thể ở xa hơn. | Giới hạn candidate trong bán kính; ETA là feature; ETA là guardrail metric. Score kết hợp nhiều mục tiêu là ablation ở Phase 5–6. |
| T2 | **Accuracy vs explainability** | Gradient boosting đủ mạnh cho dữ liệu tabular và vẫn giải thích được bằng feature contribution. Không dùng deep learning. |
| T3 | **LLM linh hoạt vs đúng đắn & chi phí** | ML xếp hạng, LLM chỉ hiểu câu hỏi và diễn đạt; số liệu luôn đến từ tool. |
| T4 | **Synthetic: kiểm soát được vs thực tế** | Synthetic cho phép biết "sự thật" để đánh giá công bằng, nhưng kết luận chỉ đúng trong giả định của simulator → ghi rõ ở Limitations. |
| T5 | **Xếp hạng từng booking vs tối ưu toàn cục** | MVP xử lý từng booking độc lập (đơn giản, latency thấp); bỏ qua xung đột khi hai booking cùng muốn một tài xế → ghi limitation, nghiên cứu ở Phase 1. |
| T6 | **Mục tiêu model chỉ là `accepted`** | Model MVP tối ưu khả năng nhận chuyến; cancellation được giám sát bằng guardrail. Kết hợp P(accept) và P(cancel) là future work / ablation. |

---

## 10. Assumptions

| ID | Assumption |
|---|---|
| A1 | Thành phố hư cấu 20 × 20 km quanh toạ độ (0, 0) với 4 hotspot; chi tiết trong [data/README.md](../data/README.md). |
| A2 | Distance là khoảng cách đường chim bay (haversine); ETA = 1 phút + distance × detour factor / tốc độ theo traffic. Không có routing thật. |
| A3 | Hành vi nhận/huỷ chuyến của tài xế tuân theo một mô hình xác suất **ẩn** trong simulator. Model ML không được truy cập mô hình này, chỉ học từ nhãn đã sample. |
| A4 | Mỗi booking được xử lý độc lập; MVP không mô phỏng việc supply tài xế cạn dần theo thời gian. |
| A5 | Offer tuần tự tối đa `N` lượt. |
| A6 | Không có PII trong dữ liệu. |

---

## 11. Risks & mitigations

| Risk | Tác động | Mitigation |
|---|---|---|
| Dữ liệu synthetic quá dễ → AUC phi thực tế | Kết luận sai | Thêm nhiễu và yếu tố ẩn không quan sát được; cảnh báo nếu AUC bất thường (H4). |
| Data leakage (ví dụ feature driver được tính từ chính các chuyến trong tập test) | Metric ảo | Split có tính thời gian; feature lịch sử chỉ tính từ dữ liệu trước thời điểm booking. |
| So sánh Baseline vs ML không công bằng | So sánh vô nghĩa | Đánh giá bằng simulator: hai policy chạy trên cùng booking test, outcome sample từ cùng mô hình ẩn với cùng random seed. |
| LLM bịa số liệu | Mất tin cậy | Bắt buộc tool cho câu hỏi dữ liệu; hiển thị tool/source; test. |
| Prompt injection qua tài liệu RAG | Assistant bị điều khiển | Tách rõ instruction và context; context chỉ là dữ liệu; test injection. |
| Chi phí LLM khi deploy public | Tốn tiền | Authentication, rate limit, max tokens; Mock provider cho dev/test/CI. |
| Scope creep | Không hoàn thành | MVP và non-goals rõ ràng; phase gate có acceptance criteria. |
| Tài nguyên máy local | Chậm | Kích thước dataset cấu hình được. Máy hiện tại (16 GB RAM, i7-12700H) ước tính đủ cho mục tiêu — kiểm chứng ở Phase 2. |

---

## 12. Open questions

| # | Câu hỏi | Chốt ở |
|---|---|---|
| Q1 | ~~Pointwise classification (P(accept)) hay Learning-to-Rank?~~ **Đã chốt:** pointwise P(accept) + XGBoost; Learning-to-Rank là ablation — xem [research.md](research.md) | Phase 1 ✅ |
| Q2 | Score cuối chỉ là P(accept) hay kết hợp cancellation/ETA? **Phase 5 đã chốt phần model:** train một model P(accept) (không train model cancellation). Phần score: so sánh A (chỉ P(accept)) và B (P(accept) với ràng buộc ETA) trên validation ở Phase 6 — xem [research §10.7](research.md) | Phase 6 |
| Q3 | "Hôm nay" nghĩa là gì với dữ liệu synthetic tĩnh (ngày mới nhất trong dataset, hay sinh dữ liệu tương đối theo ngày hiện tại)? | Phase 8, 10 |
| Q4 | Matching chạy từ UI thì outcome (accepted/cancelled) đến từ đâu khi không có tài xế thật: mô phỏng bằng simulator hay chỉ lưu recommendation? | Phase 8–9 |
| Q5 | Authentication: API key tĩnh hay JWT login cho admin? | Phase 9 |
| Q6 | Vector store: pgvector hay Chroma (đề xuất: pgvector)? | Phase 11 |
| Q7 | Commit model artifact vào git hay train trong bước build? | Phase 7, 15 |
| Q8 | ~~Training labels chỉ có cho tài xế đã được offer hay có cho mọi candidate?~~ **Đã chốt:** cả hai — log thực tế (nearest-first + 20% exploration) cho training; `data/oracle/` có outcome cho mọi candidate, chỉ dành cho evaluation — xem [data/README.md](../data/README.md) | Phase 2 ✅ |
| Q9 | ~~Có thêm baseline phụ (weighted rule score) và dòng Oracle (upper bound từ mô hình ẩn) vào evaluation không?~~ **Đã chốt:** có, cùng Random làm mức sàn — xem [evaluation.md](evaluation.md) | Phase 4 ✅ |

---

## 13. Roadmap

| Phase | Tên | Deliverable chính |
|---|---|---|
| 0 | Product Definition | PRD, architecture, repository skeleton |
| 1 | Research | `docs/research.md` |
| 2 | Synthetic Dataset | Generator + `data/README.md` |
| 3 | EDA | `notebooks/01_eda.ipynb`, findings trong `docs/research.md` |
| 4 | Baseline | Nearest Driver + baseline metrics |
| 5 | Train ML Model | Model đã train + tuning |
| 6 | Evaluation & Benchmarking | `docs/evaluation.md` |
| 7 | Model Serving | `ml/inference/predict.py` |
| 8 | PostgreSQL | ORM models, Alembic migrations, seed |
| 9 | FastAPI | API + `docs/api.md` |
| 10 | LLM | Provider abstraction + assistant cơ bản |
| 11 | RAG | `knowledge/` + `backend/app/rag/` |
| 12 | AI Agent | LangGraph agent + tools |
| 13 | Guardrails | Scope, injection, data protection, tool validation |
| 14 | Frontend | Dashboard, Matching Simulator, Chat |
| 15 | Docker | Dockerfiles + `docker-compose.yml` |
| 16 | Testing | Lấp khoảng trống test, coverage ≥ 70% |
| 17 | Monitoring | Structured logs, latency P50/P95/P99 |
| 18 | Deployment | VPS + Nginx + HTTPS, `docs/deployment.md` |
| 19 | CI/CD | GitHub Actions |
| 20 | Final Experiment | `docs/experiment.md`, README hoàn chỉnh |

Test được viết **trong từng phase** cùng với code; Phase 16 là để đo coverage và bổ sung phần còn thiếu, không phải lúc bắt đầu viết test.

---

## 14. Glossary

| Thuật ngữ | Nghĩa trong project |
|---|---|
| Booking | Yêu cầu chuyến đi của hành khách. |
| Candidate | Tài xế available nằm trong bán kính tìm kiếm của booking. |
| Offer | Lượt hệ thống gửi booking cho một tài xế. |
| ETA | Thời gian ước tính để tài xế tới điểm đón. |
| Policy | Cách chọn/xếp hạng tài xế (Nearest Driver, ML ranking). |
| Baseline | Policy đơn giản làm mốc so sánh. |
| Data leakage | Model vô tình dùng thông tin không có tại thời điểm dự đoán. |
| Guardrail metric | Metric không được phép xấu đi khi tối ưu North Star metric. |
| Tool calling | LLM yêu cầu gọi một hàm định nghĩa sẵn thay vì tự trả lời. |
| RAG | Retrieval-Augmented Generation — trả lời dựa trên tài liệu được truy xuất. |
