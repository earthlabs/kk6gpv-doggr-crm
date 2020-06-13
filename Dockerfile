FROM python:3.7-slim-buster

LABEL maintainer="areed145@gmail.com"

WORKDIR /doggr-crm

COPY . /doggr-crm

# We copy just the requirements.txt first to leverage Docker cache
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENV MONGODB_CLIENT 'mongodb+srv://kk6gpv:kk6gpv@cluster0-kglzh.azure.mongodb.net/test?retryWrites=true&w=majority'

CMD ["python", "doggr-crm.py"]
