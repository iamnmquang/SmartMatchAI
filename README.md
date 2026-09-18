# SmartMatch AI

Mini production AI product (mô phỏng): **xếp hạng tài xế cho booking bằng Machine Learning**, kèm **AI Assistant**
(LangGraph agent + tool calling + RAG) để hỏi đáp về matching, số liệu và policy của hệ thống.

> **Disclaimer.** Đây là project học tập. Dữ liệu 100% synthetic. Project không sử dụng dữ liệu, không mô phỏng
> kiến trúc hay logic production của bất kỳ công ty nào.

## Status

**Phase 6 — Evaluation & Benchmarking: hoàn thành.** Trên tập test, ML ranking đạt Matching Success Rate **86.5%**
(baseline Nearest Driver 85.6%) và Completed Rate **75.9%** (75.5%), với ETA chỉ tăng 2.0% —
chi tiết và giới hạn ở [evaluation.md](docs/evaluation.md). README đầy đủ (screenshots, demo) sẽ được viết ở Phase 20.

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
| [Research](docs/research.md) | Problem formulation, related work, so sánh 6 hướng matching, chọn baseline và ML approach, experiment design, EDA findings, kết quả training | 1, 3, 5 |
| [Data](data/README.md) | Data policy, cách generate, schema, mô hình sinh dữ liệu, thống kê | 0, 2 |
| [EDA notebook](notebooks/01_eda.ipynb) | Chất lượng dữ liệu, phân phối, outlier, confounding, selection bias, feature importance sơ bộ | 3 |
| [Evaluation](docs/evaluation.md) | Protocol đánh giá offline, định nghĩa metric, chọn policy trên validation, **ML vs Baseline trên test**, ML metrics, ablation, kết luận giả thuyết | 4, 6 |
| [Master prompt](docs/master-prompt.md) | Yêu cầu gốc của project (nguyên văn) — đối chiếu phạm vi, quy tắc và quy trình theo phase | — |
| [Worklog](worklog-overview/) | Tổng kết công việc theo ngày — [2026-09-15](worklog-overview/2026-09-15.md): Phase 0–4 | — |

## Roadmap

| Phase | Tên | Status |
|---|---|---|
| 0 | Product Definition | ✅ Done |
| 1 | Research | ✅ Done |
| 2 | Synthetic Dataset | ✅ Done |
| 3 | EDA | ✅ Done |
| 4 | Baseline (Nearest Driver) | ✅ Done |
| 5 | Train ML Model | ✅ Done |
| 6 | Evaluation & Benchmarking | ✅ Done |
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
- GNU Make (tuỳ chọn, khi Makefile được thêm; trên Windows cài qua Chocolatey/Scoop hoặc dùng WSL)

## Setup

```bash
cp .env.example .env                 # điền giá trị; không bao giờ commit .env

python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## Chạy pipeline

```bash
python -m ml.data.generate           # Phase 2: sinh dữ liệu synthetic → data/raw/, data/oracle/
pytest                               # chạy test

# Phase 3: chạy lại EDA từ đầu đến cuối (hoặc mở notebook bằng Jupyter / VS Code với kernel .venv)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

python -m ml.evaluation.run_baseline # Phase 4: đánh giá baseline → ml/evaluation/results/baseline.json
python -m ml.training.train          # Phase 5: train P(accept) → ml/training/results/training.json

# Phase 6: chọn policy trên validation, đo một lần trên test; ablation feature
python -m ml.evaluation.run_evaluation
python -m ml.evaluation.run_ablations
```

Các bước tiếp theo sẽ được bổ sung theo từng phase.
