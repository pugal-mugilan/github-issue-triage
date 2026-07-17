FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY models/encoders.pkl models/encoders.pkl
COPY models/scaler.pkl models/scaler.pkl
COPY models/nn_weighted_model.pt models/nn_weighted_model.pt

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]