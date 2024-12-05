FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y gcc g++ build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /src
RUN mkdir /tests
WORKDIR /src

ENV PATH="${PATH}:/root/.local/bin"
ENV PYTHONPATH=.
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip --default-timeout=300 install -r requirements.txt

COPY src/ src/
COPY tests/ tests/

RUN pip install pytest

# EXPOSE 8000
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
CMD ["python", "-m", "RAGBench.main"]