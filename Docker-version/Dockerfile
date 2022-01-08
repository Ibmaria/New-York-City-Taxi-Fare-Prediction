FROM python:3.7.9
RUN mkdir /app
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
RUN chmod +x entrypoint.sh
EXPOSE 8000  


# start server  
# 
#command: bash -c "python manage.py makemigrations &&
#                    python manage.py migrate"
#CMD [ "python", "manage.py", "runserver", "0.0.0.0:8000" ]entrypoint: ./entrypoint.sh