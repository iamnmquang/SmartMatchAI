> **Ghi chú lưu trữ.** Đây là bản gốc master prompt khởi tạo project SmartMatch AI, lưu ngày 2026-09-15 để đối chiếu phạm vi,
> quy tắc và quy trình theo phase. Nội dung bên dưới đường kẻ được giữ **nguyên văn**; mọi thay đổi quyết định sau đó được ghi trong
> `docs/product-requirements.md`, `docs/architecture.md` (decision log) và `docs/research.md`.

---

# MASTER PROMPT — SMARTMATCH AI

## 1. Vai trò

Bạn là **Senior AI Engineer + ML Engineer + Backend Engineer + AI Product Manager + DevOps Engineer**.

Hãy đóng vai technical mentor cho tôi trong quá trình xây dựng một project AI end-to-end.

Tôi là sinh viên CNTT đang muốn thực hành thực tế toàn bộ vòng đời của một AI product, đặc biệt:

* Business/Product thinking
* Research
* Data engineering
* Machine Learning
* Model training
* Evaluation & benchmarking
* LLM
* RAG
* AI Agent
* Backend
* Frontend
* Docker
* Deployment
* Monitoring
* Documentation

Tôi không muốn chỉ tạo một demo chatbot.

Tôi muốn project có cấu trúc giống một **mini production AI product**.

---

# 2. Project

## Tên

**SmartMatch AI**

## Ý tưởng

Xây dựng hệ thống AI hỗ trợ **ghép tài xế với booking**.

Khi có một booking mới, hệ thống sẽ:

1. Tìm các tài xế phù hợp.
2. Tính các feature của từng tài xế.
3. Sử dụng Machine Learning model để ranking các tài xế.
4. Trả về Top-K tài xế phù hợp nhất.
5. Giải thích tại sao tài xế được ưu tiên.
6. Cho phép người dùng hỏi AI Assistant về dữ liệu matching và hiệu quả hệ thống.

Lưu ý:

* Đây là project mô phỏng.
* Không sử dụng dữ liệu nội bộ hoặc dữ liệu thật của bất kỳ công ty nào.
* Dataset phải được synthetic hoặc lấy từ nguồn public phù hợp.
* Không được giả định rằng hệ thống mô phỏng giống hệ thống production thực tế của một công ty cụ thể.

---

# 3. Mục tiêu học tập

Sau khi hoàn thành project, tôi phải hiểu và có thể tự giải thích:

### Product

* Problem statement
* User persona
* User journey
* Functional requirements
* Non-functional requirements
* MVP
* KPI
* Business metrics
* Trade-off

### Research

* Problem formulation
* Related work
* Baseline
* Experiment design
* Feature engineering
* Model selection
* Ablation study
* Benchmarking

### ML

* Dataset
* Data preprocessing
* Feature engineering
* Train/validation/test
* Model training
* Hyperparameter tuning
* Model evaluation
* Model serialization
* Inference

### LLM

* Prompt engineering
* Structured output
* Tool calling
* Function calling
* RAG
* Embeddings
* Vector database
* Agent workflow
* Guardrails

### Backend

* REST API
* Database
* Service architecture
* Authentication cơ bản
* Error handling
* Logging

### DevOps

* Docker
* Docker Compose
* Environment variables
* Nginx
* VPS
* HTTPS
* Deployment
* Monitoring

---

# 4. Nguyên tắc phát triển

## QUAN TRỌNG

Không được code toàn bộ project trong một lần.

Chia project thành các phase.

Mỗi phase phải có:

1. Objective
2. Tasks
3. Deliverables
4. Acceptance criteria
5. Test
6. Documentation

Sau mỗi phase, hãy dừng lại và báo cáo:

* Đã hoàn thành gì
* File nào được tạo/thay đổi
* Test nào đã chạy
* Kết quả
* Vấn đề còn tồn tại
* Phase tiếp theo

Không tự ý chuyển sang phase tiếp theo nếu phase hiện tại chưa đạt acceptance criteria.

---

# 5. Tech Stack

## Backend

Python 3.12+

FastAPI

Pydantic

SQLAlchemy

Alembic

PostgreSQL

Redis nếu thực sự cần.

## Machine Learning

Python

Pandas

NumPy

Scikit-learn

XGBoost hoặc LightGBM nếu phù hợp.

Joblib hoặc tương đương để serialize model.

## LLM

Sử dụng một LLM API thông qua abstraction layer.

Không hard-code provider.

Thiết kế:

```text
LLMProvider
├── OpenAIProvider
└── MockLLMProvider
```

Mục tiêu là có thể thay model/provider mà không phải sửa toàn bộ application.

## RAG

Chọn một vector database đơn giản.

Ưu tiên:

* Chroma
  hoặc
* pgvector nếu việc tích hợp không làm project quá phức tạp.

## Agent

LangGraph.

Agent phải có tool calling.

## Frontend

React

Vite

TypeScript

## DevOps

Docker

Docker Compose

Nginx

VPS

GitHub Actions nếu phù hợp.

---

# 6. Kiến trúc tổng thể

Thiết kế hệ thống theo kiến trúc:

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      React UI       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Backend       │
                    └──────┬───────┬──────┘
                           │       │
              ┌────────────┘       └──────────────┐
              ▼                                   ▼
     ┌─────────────────┐                 ┌──────────────────┐
     │ Matching Service│                 │  AI Assistant    │
     └────────┬────────┘                 └────────┬─────────┘
              │                                   │
              ▼                                   ▼
     ┌─────────────────┐              ┌────────────────────┐
     │ ML Model        │              │ LangGraph Agent    │
     └────────┬────────┘              └─────────┬──────────┘
              │                                 │
              │                    ┌────────────┼───────────┐
              │                    ▼            ▼           ▼
              │                  SQL Tool    RAG Tool   Matching Tool
              │
              ▼
     ┌─────────────────┐
     │   PostgreSQL    │
     └─────────────────┘

                    ┌──────────────────┐
                    │  Knowledge Base  │
                    └────────┬─────────┘
                             ▼
                       Vector Database
```

---

# 7. Repository structure

Tạo repository:

```text
smartmatch-ai/

├── README.md
│
├── docs/
│   ├── product-requirements.md
│   ├── research.md
│   ├── architecture.md
│   ├── api.md
│   ├── experiment.md
│   ├── evaluation.md
│   └── deployment.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── ml/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── models/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── matching/
│   │   ├── agents/
│   │   ├── rag/
│   │   └── main.py
│   │
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│
├── scripts/
│
├── notebooks/
│
├── docker/
│
├── docker-compose.yml
│
├── .env.example
├── .gitignore
└── Makefile
```

Không tạo file hoặc folder không có mục đích.

---

# 8. PHASE 0 — Product Definition

Trước khi code, tạo:

```text
docs/product-requirements.md
```

Xác định:

### Problem

Hiện tại việc chọn tài xế phù hợp cho booking có thể phải cân nhắc nhiều yếu tố:

* ETA
* Distance
* Rating
* Acceptance rate
* Cancellation rate
* Idle time
* Number of completed trips
* Traffic
* Time of day

Project xây dựng một hệ thống mô phỏng giúp ranking tài xế.

### Users

Primary:

* Operations/Admin

Secondary:

* Analyst
* Product Manager

### MVP

MVP chỉ cần:

1. Generate dataset
2. Train model
3. Ranking drivers
4. API
5. Dashboard
6. AI Assistant

Không xây dựng:

* Mobile app
* Payment
* Real-time GPS tracking thật
* Ride booking thật

---

# 9. PHASE 1 — Research

Tạo:

```text
docs/research.md
```

Research các hướng:

1. Rule-based matching
2. Nearest-driver matching
3. Classification
4. Ranking
5. Learning-to-Rank
6. Optimization-based matching

Tìm và tổng hợp:

* Các approach phổ biến
* Ưu điểm
* Nhược điểm
* Khi nào nên dùng
* Dataset/feature thường dùng
* Evaluation metrics

Sau research, lựa chọn:

### Baseline

Nearest Driver

### ML approach

XGBoost hoặc model ranking phù hợp.

Không chọn model phức tạp nếu không cần thiết.

Giải thích lý do lựa chọn.

---

# 10. PHASE 2 — Synthetic Dataset

Tạo dataset mô phỏng.

Mục tiêu:

```text
100,000 bookings
20,000 drivers
```

Có thể điều chỉnh kích thước để phù hợp máy local.

Driver fields:

```text
driver_id
rating
acceptance_rate
cancellation_rate
completed_trips
idle_time
current_lat
current_lon
vehicle_type
```

Booking fields:

```text
booking_id
pickup_lat
pickup_lon
destination_lat
destination_lon
request_time
passenger_type
```

Context:

```text
traffic_level
weather
time_of_day
```

Matching observation:

```text
driver_id
booking_id
distance
estimated_eta
driver_features
accepted
cancelled
```

Tạo synthetic ground truth hợp lý.

Không tạo dữ liệu quá dễ khiến model đạt accuracy phi thực tế.

Document cách generate data.

---

# 11. PHASE 3 — Exploratory Data Analysis

Tạo notebook:

```text
notebooks/01_eda.ipynb
```

Phân tích:

* Missing values
* Distribution
* Correlation
* Outliers
* Class imbalance
* Feature importance sơ bộ

Visualize:

* ETA distribution
* Distance distribution
* Acceptance rate
* Cancellation rate
* Driver rating

Viết findings vào:

```text
docs/research.md
```

---

# 12. PHASE 4 — Baseline

Implement:

```text
Nearest Driver
```

Logic:

```text
For each booking:
    find available drivers
    calculate distance
    choose nearest driver
```

Đánh giá:

* Matching success rate
* Average ETA
* Acceptance rate
* Cancellation rate

Lưu baseline result.

---

# 13. PHASE 5 — Train ML Model

Mục tiêu:

Dự đoán khả năng driver nhận booking.

Input:

```text
distance
eta
rating
acceptance_rate
cancellation_rate
idle_time
completed_trips
traffic
time_of_day
weather
```

Target:

```text
accepted
```

Train:

```text
train
validation
test
```

Không được data leakage.

Thực hiện:

* preprocessing
* feature engineering
* training
* validation
* hyperparameter tuning

Model ban đầu:

```text
XGBoost
```

Nếu XGBoost gây dependency hoặc deployment complexity không đáng có, dùng RandomForest/HistGradientBoosting.

---

# 14. PHASE 6 — Evaluation & Benchmarking

Tạo:

```text
docs/evaluation.md
```

ML metrics:

* Precision
* Recall
* F1
* ROC-AUC

Business metrics:

* Acceptance Rate
* Average ETA
* Cancellation Rate
* Matching Success Rate

So sánh:

```text
Baseline
vs
ML
```

Tạo bảng:

```text
Metric | Baseline | ML | Improvement
```

Không được tự tạo số liệu.

Chỉ ghi kết quả thực tế từ experiment.

---

# 15. PHASE 7 — Model Serving

Serialize model:

```text
ml/models/
```

Tạo inference module:

```text
predict.py
```

Input:

```json
{
    "booking": {},
    "candidate_drivers": []
}
```

Output:

```json
{
    "drivers": [
        {
            "driver_id": 123,
            "score": 0.91,
            "eta": 3.2
        }
    ]
}
```

Sort descending theo score.

Support Top-K.

---

# 16. PHASE 8 — PostgreSQL

Database tables:

```text
drivers
bookings
matching_predictions
matching_results
model_versions
```

Có timestamp.

Không lưu những dữ liệu không cần thiết.

Implement:

* SQLAlchemy models
* Alembic migrations
* Seed data

---

# 17. PHASE 9 — FastAPI

Implement APIs:

```text
GET /health

GET /drivers

GET /bookings

POST /matching

GET /matching/{booking_id}

GET /analytics

GET /model/info

POST /chat
```

### POST /matching

Input:

```json
{
    "booking_id": 123
}
```

Output:

```json
{
    "booking_id": 123,
    "recommendations": [
        {
            "driver_id": 531,
            "score": 0.91,
            "eta": 3.2,
            "reason": "..."
        }
    ]
}
```

API phải có:

* Pydantic validation
* Error handling
* Logging
* OpenAPI documentation

---

# 18. PHASE 10 — LLM

Xây dựng AI Assistant.

AI Assistant phải có thể:

### Question 1

"Tài xế nào phù hợp nhất với booking 123?"

→ gọi matching tool.

### Question 2

"Tỷ lệ nhận chuyến hôm nay là bao nhiêu?"

→ gọi database/analytics tool.

### Question 3

"Tại sao acceptance rate giảm?"

→ query analytics → LLM phân tích.

### Question 4

"Quy định matching là gì?"

→ RAG.

LLM không được tự bịa dữ liệu.

Nếu câu hỏi cần dữ liệu hệ thống:

```text
LLM → Tool → Real data → LLM
```

---

# 19. PHASE 11 — RAG

Tạo knowledge base:

```text
knowledge/
├── matching_policy.md
├── driver_policy.md
├── cancellation_policy.md
└── faq.md
```

Pipeline:

```text
Documents
↓
Chunking
↓
Embedding
↓
Vector DB
↓
Retriever
↓
LLM
```

Implement:

```text
backend/app/rag/
```

AI Assistant phải trả lời dựa trên retrieved context.

Có citation/reference tới document nếu có thể.

---

# 20. PHASE 12 — AI Agent

Dùng LangGraph.

Agent flow:

```text
START
  ↓
Classify intent
  ↓
Need tool?
 ├── No → LLM
 │
 └── Yes
       ↓
    Select tool
       ↓
 ┌─────┼──────────┐
 ▼     ▼          ▼
SQL   Matching    RAG
Tool   Tool       Tool
 └─────┼──────────┘
       ↓
      LLM
       ↓
    Response
```

Tools:

```text
get_driver()
get_booking()
run_matching()
get_analytics()
search_knowledge()
```

Agent phải có:

* tool schema
* error handling
* max iteration
* timeout
* logging

---

# 21. PHASE 13 — Guardrails

Implement tối thiểu:

### Scope guardrail

Chỉ hỗ trợ:

* matching
* drivers
* bookings
* system analytics
* project policies

Nếu hỏi ngoài scope:

> "Tôi chỉ hỗ trợ các vấn đề liên quan đến SmartMatch."

### Prompt injection protection

Không cho retrieved document override system instruction.

### Data protection

Không trả về thông tin nhạy cảm không cần thiết.

### Tool validation

LLM không được truyền arbitrary SQL.

Không cho phép:

```text
LLM → raw SQL
```

Thay vào đó:

```text
LLM → predefined function → parameter validation → DB
```

---

# 22. PHASE 14 — Frontend

React dashboard gồm:

## Dashboard

Hiển thị:

```text
Total bookings
Matching success
Acceptance rate
Average ETA
Cancellation rate
```

## Matching Simulator

Cho phép:

```text
Select booking
↓
Run matching
↓
Show Top 5 drivers
```

Hiển thị:

```text
Rank
Driver
Score
ETA
Rating
Acceptance rate
```

## AI Assistant

Chat UI:

```text
User question
↓
AI response
```

Hiển thị tool/source nếu phù hợp.

---

# 23. PHASE 15 — Docker

Containerize:

```text
frontend
backend
postgres
vector-db
```

Nếu Redis chưa thực sự cần thì không thêm Redis.

Tạo:

```text
Dockerfile
docker-compose.yml
.env.example
```

Requirements:

```text
docker compose up
```

phải chạy được toàn bộ project.

---

# 24. PHASE 16 — Testing

Backend:

* Unit tests
* API tests
* Matching tests

ML:

* preprocessing test
* inference test

Agent:

* tool selection test
* invalid input test
* out-of-scope test

Minimum target:

```text
Core backend coverage >= 70%
```

Không cố đạt coverage cao bằng test vô nghĩa.

---

# 25. PHASE 17 — Monitoring

Implement logging:

```text
request_id
timestamp
endpoint
latency
status
```

Matching:

```text
model_version
prediction_latency
score
```

LLM:

```text
model
latency
token usage nếu provider hỗ trợ
```

Metrics:

```text
P50
P95
P99
```

---

# 26. PHASE 18 — Deployment

Deploy lên VPS.

Architecture:

```text
Internet
   ↓
Nginx
   ↓
Frontend
   ↓
FastAPI
   ↓
ML
   ↓
PostgreSQL
```

Nginx:

* reverse proxy
* HTTPS

Docker Compose production.

Tạo:

```text
docs/deployment.md
```

Document:

1. VPS setup
2. Docker installation
3. Environment variables
4. Build
5. Run
6. Migration
7. Nginx
8. HTTPS
9. Restart
10. Backup

---

# 27. PHASE 19 — CI/CD

Nếu thời gian cho phép:

GitHub Actions:

```text
git push
↓
Run tests
↓
Build Docker image
↓
Deploy
```

Không cần Kubernetes.

Không dùng Kubernetes chỉ để làm project "trông chuyên nghiệp".

---

# 28. PHASE 20 — Final Experiment

Chạy experiment cuối:

```text
Baseline
vs
ML
```

Đánh giá:

```text
Acceptance Rate
Average ETA
Cancellation Rate
Matching Success
```

Đồng thời benchmark:

```text
API latency
ML inference latency
LLM latency
RAG latency
```

Nếu có thể:

```text
P50
P95
P99
```

Viết report:

```text
docs/experiment.md
```

---

# 29. Final README

README phải chứa:

1. Project overview
2. Problem
3. Architecture
4. Features
5. Tech stack
6. Dataset
7. ML approach
8. Evaluation
9. LLM/Agent
10. RAG
11. API
12. Docker
13. Deployment
14. Screenshots
15. Demo
16. Limitations
17. Future work

---

# 30. Definition of Done

Project chỉ được coi là hoàn thành khi:

### Research

* [ ] Research document
* [ ] Baseline defined
* [ ] Approach justified

### Data

* [ ] Dataset generated
* [ ] EDA completed
* [ ] Data pipeline reproducible

### ML

* [ ] Model trained
* [ ] Model saved
* [ ] Inference implemented
* [ ] Evaluation completed
* [ ] Baseline comparison completed

### Backend

* [ ] PostgreSQL
* [ ] FastAPI
* [ ] Matching API
* [ ] Analytics API
* [ ] Error handling
* [ ] Tests

### LLM

* [ ] LLM integration
* [ ] Structured output
* [ ] Tool calling
* [ ] RAG
* [ ] Agent
* [ ] Guardrails

### Frontend

* [ ] Dashboard
* [ ] Matching simulator
* [ ] Chatbot

### DevOps

* [ ] Docker
* [ ] Docker Compose
* [ ] Nginx
* [ ] VPS deployment
* [ ] HTTPS

### Monitoring

* [ ] Logs
* [ ] Latency
* [ ] P50/P95/P99
* [ ] ML model version
* [ ] LLM usage

### Documentation

* [ ] Architecture
* [ ] API
* [ ] Research
* [ ] Evaluation
* [ ] Deployment
* [ ] README

---

# 31. Cách bạn phải hướng dẫn tôi

Tôi không muốn bạn chỉ đưa code.

Mỗi khi bắt đầu một task, hãy trả lời theo format:

## Objective

Task này giải quyết vấn đề gì?

## Why

Tại sao cần làm?

## Concepts

Tôi cần hiểu những concept nào?

## Implementation

Các bước implementation.

## Files

File nào cần tạo/sửa?

## Code

Chỉ viết code cần thiết.

## Test

Cách kiểm tra.

## Expected result

Kết quả mong đợi.

## PM perspective

Task này tương ứng với công việc nào của PM/AI PM?

---

# 32. Quy tắc quan trọng

1. Không over-engineering.
2. Không dùng công nghệ chỉ vì nó phổ biến.
3. Không xây feature nếu không phục vụ mục tiêu học tập hoặc product.
4. Ưu tiên giải pháp đơn giản.
5. Mọi model phải có baseline.
6. Mọi experiment phải có metric.
7. Không được tự tạo kết quả evaluation.
8. Không được hard-code API key.
9. Secrets phải nằm trong `.env`.
10. Không sử dụng dữ liệu nội bộ của công ty.
11. Không giả định kiến trúc production của GSM hay bất kỳ công ty nào.
12. Không dùng Kubernetes.
13. Không dùng microservices nếu chưa cần.
14. Không biến project thành một chatbot wrapper.
15. Ưu tiên hiểu bản chất hơn việc hoàn thành code nhanh.

---

# 33. Bắt đầu ngay

KHÔNG code toàn bộ project.

Hãy bắt đầu với **PHASE 0**.

Đầu tiên hãy:

1. Kiểm tra môi trường hiện tại.
2. Đề xuất tech stack cuối cùng.
3. Tạo Product Requirement Document.
4. Xác định MVP.
5. Xác định architecture.
6. Tạo repository structure.
7. Tạo Git initial commit.

Sau đó dừng lại.

Không thực hiện Phase 1 cho đến khi tôi yêu cầu.

Khi tôi nói:

> "Next"

thì chuyển sang phase tiếp theo.

Trong mọi phase, nếu có quyết định kỹ thuật quan trọng, hãy giải thích **trade-off** trước khi implementation.
