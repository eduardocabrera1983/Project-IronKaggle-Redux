# 1) Base image
FROM python:3.9-slim

# 2) Set workdir
WORKDIR /app

# 3) Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy code & model
COPY app.py .
COPY model/ ./model/

# 5) Expose port & run with gunicorn
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
