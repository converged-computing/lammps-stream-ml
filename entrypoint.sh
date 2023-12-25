#!/bin/bash

url=${1:-"0.0.0.0:8080"}
echo "Will deploy to ${url}"

# Note that this is not a production setup, just for prototype
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver ${url}
