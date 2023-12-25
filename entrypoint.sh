#!/bin/bash

# Note that this is not a production setup, just for prototype
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver
