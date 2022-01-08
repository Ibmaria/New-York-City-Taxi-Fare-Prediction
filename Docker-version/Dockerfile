FROM python:3.7.9
RUN mkdir /app
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
RUN chmod +x entrypoint.sh
EXPOSE 8000  


