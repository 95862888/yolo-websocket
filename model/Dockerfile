FROM ultralytics/ultralytics:latest
COPY .. /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python3", "-u", "main.py"]
