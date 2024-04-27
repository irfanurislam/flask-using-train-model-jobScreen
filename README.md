Using Libraries --------
from app import app, mysql, models
from flask import render_template, request, session, redirect, url_for
import mysql.connector
from mysql.connector import Error
from app.models import create_job_offer_table, insert_job_list
from werkzeug.security import generate_password_hash, check_password_hash
