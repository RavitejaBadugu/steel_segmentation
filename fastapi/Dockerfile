FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /fastapi

COPY fastapi_requriements.txt /fastapi

WORKDIR /fastapi

RUN pip install -r fastapi_requriements.txt

COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]