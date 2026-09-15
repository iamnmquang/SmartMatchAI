# SmartMatch AI

Mini production AI product (mô phỏng): **xếp hạng tài xế cho booking bằng Machine Learning**, kèm **AI Assistant**
(LangGraph agent + tool calling + RAG) để hỏi đáp về matching, số liệu và policy của hệ thống.

> **Disclaimer.** Đây là project học tập. Dữ liệu 100% synthetic. Project không sử dụng dữ liệu, không mô phỏng
> kiến trúc hay logic production của bất kỳ công ty nào.

## Status

**Phase 1 — Research: hoàn thành.** README đầy đủ (kết quả, screenshots, demo) sẽ được viết ở Phase 20.

## Ý tưởng

```text
New booking
   │
   ▼
Candidate drivers (trong bán kính)
   │
   ▼
Features: distance, ETA, rating, acceptance/cancellation rate, idle time, completed trips, traffic, time of day, weather
   │
   ▼
ML model scoring  ──►  Top-K drivers + reason
                          │
                          ▼
         Dashboard  ·  Matching Simulator  ·  AI Assistant
```

Baseline để so sánh: **Nearest Driver**. ML chỉ được coi là tốt hơn khi thắng baseline trên business metrics
mà không làm xấu guardrail metrics (xem [PRD §8](docs/product-requirements.md#8-metrics--kpis)).

## Documentation

| Tài liệu | Nội dung | Phase |
|---|---|---|
| [Product Requirements](docs/product-requirements.md) | Problem, personas, user journeys, MVP, FR/NFR, KPIs, trade-offs, risks | 0 |
| [Architecture](docs/architecture.md) | Components, flows, repository structure, tech stack, decision log | 0 |
| [Research](docs/research.md) | Problem formulation, related work, so sánh 6 hướng matching, chọn baseline và ML approach, experiment design | 1 |
| [Data policy](data/README.md) | Nguồn dữ liệu, quy tắc, cách generate | 0, 2 |

## Roadmap

| Phase | Tên | Status |
|---|---|---|
| 0 | Product Definition | ✅ Done |
| 1 | Research | ✅ Done |
| 2 | Synthetic Dataset | ⬜ |
| 3 | EDA | ⬜ |
| 4 | Baseline (Nearest Driver) | ⬜ |
| 5 | Train ML Model | ⬜ |
| 6 | Evaluation & Benchmarking | ⬜ |
| 7 | Model Serving | ⬜ |
| 8 | PostgreSQL | ⬜ |
| 9 | FastAPI | ⬜ |
| 10 | LLM | ⬜ |
| 11 | RAG | ⬜ |
| 12 | AI Agent (LangGraph) | ⬜ |
| 13 | Guardrails | ⬜ |
| 14 | Frontend | ⬜ |
| 15 | Docker | ⬜ |
| 16 | Testing | ⬜ |
| 17 | Monitoring | ⬜ |
| 18 | Deployment | ⬜ |
| 19 | CI/CD | ⬜ |
| 20 | Final Experiment | ⬜ |

## Tech stack (tóm tắt)

Python 3.12 · FastAPI · Pydantic · SQLAlchemy · Alembic · PostgreSQL · pandas · scikit-learn · XGBoost ·
LLM provider abstraction (OpenAI / Mock) · LangGraph · pgvector (proposed) · React · Vite · TypeScript ·
Docker Compose · Nginx · GitHub Actions.

Lý do lựa chọn và trade-off: [Architecture §8](docs/architecture.md#8-tech-stack).

## Prerequisites

- Python 3.12+
- Node.js 22+
- Docker + Docker Compose v2
- Git
- GNU Make (từ Phase 2; trên Windows cài qua Chocolatey/Scoop hoặc dùng WSL)

## Setup

```bash
cp .env.example .env   # điền giá trị; không bao giờ commit .env
```

Các bước chạy project sẽ được bổ sung theo từng phase.
