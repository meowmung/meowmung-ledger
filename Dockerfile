# Dockerfile

# For application_connect_RESTful_API.py

FROM python:3.9.20-slim

WORKDIR /meowmung_ledger_docker

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8085

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8085/health || exit 1

CMD ["uvicorn", "application_connect_RESTful_API:app", "--host", "0.0.0.0", "--port", "8085"]


# # For application_connect_gRPC_API.py

# FROM python:3.9.20-slim

# WORKDIR /meowmung_ledger_docker

# COPY requirements.txt .

# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8085

# HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
#     CMD curl -f http://localhost:8085/health || exit 1

# CMD ["uvicorn", "application_connect_gRPC_API:app", "--host", "0.0.0.0", "--port", "8085"]