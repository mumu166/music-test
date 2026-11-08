FROM mirror.ccs.tencentyun.com/library/python:3.9-slim

WORKDIR /app

# 腾讯云 pip 镜像加速
RUN pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple && \
    pip config set global.trusted-host mirrors.cloud.tencent.com

# 只装 ffmpeg 本体，不装推荐依赖（关键：--no-install-recommends）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 装 torch CPU 版（合并一步，清缓存）
RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# demucs 单独装，不拉它的依赖（torch 已经有了）
RUN pip install --no-cache-dir --no-deps "demucs>=4.0.1"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]