FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get -y update
RUN apt-get -y install curl
RUN apt-get -y install vim


RUN pip install --no-cache-dir --upgrade pip
COPY ./ /code
RUN pip install --no-cache-dir /code
RUN rm -rf /code

# Create the directory where the uploaded files will be stored
RUN mkdir -p /app/tmp && chmod -R 777  /app/tmp

# Change working directory since uvicorn does not allow for paths with `.` inside
WORKDIR /usr/local/lib/python3.10

EXPOSE 8000
ENTRYPOINT uvicorn $(whyhow-locate) --host 0.0.0.0 --port 8000
