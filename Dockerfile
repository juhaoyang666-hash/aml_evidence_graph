FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY src ./src
COPY knowledge ./knowledge

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" ".[llm]"

ENV AML_ENV=container
ENV AML_LLM_ENABLED=false
ENV AML_TYPOLOGY_ROOT=/app/knowledge/typologies

EXPOSE 8000
CMD ["uvicorn", "aml_evidence_graph.api.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
