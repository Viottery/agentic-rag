FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /workspace

RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org/debian|https://mirrors.nju.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources && \
        sed -i 's|http://security.debian.org/debian-security|https://mirrors.nju.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ripgrep \
    build-essential \
    clinfo \
    libdrm-intel1 \
    libigdgmm12 \
    libze1 \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -i ${PIP_INDEX_URL} -r /tmp/requirements.txt

CMD ["fastapi", "dev", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
