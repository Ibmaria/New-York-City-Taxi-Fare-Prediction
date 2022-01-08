#!/bin/sh
python manage.py makemigrations taxi
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
exec "$@"