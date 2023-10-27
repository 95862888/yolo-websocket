FROM python:3.10.13-slim
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python3", "-u", "main.py"]
