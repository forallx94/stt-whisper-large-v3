# UI 는 전사를 하지 않는다. API 를 HTTP 로 부르기만 하므로 torch 도 CUDA 도 필요 없다.
# 그래서 본체 이미지(12GB)와 따로 만든다 -- 이쪽은 수백 MB 로 끝나고 빌드도 빠르다.
FROM python:3.12-slim

COPY requirements.txt /ui/requirements.txt
RUN pip install --no-cache-dir -r /ui/requirements.txt

COPY app.py /ui/app.py

ENV PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /ui
ENTRYPOINT ["python", "/ui/app.py"]
