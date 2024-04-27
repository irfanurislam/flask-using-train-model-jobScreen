import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from app import app, mysql


def insert_job_list(job_title, company_name, about_company, company_address, job_description, skills, skill_categories, key_responsibility, qualification, benefits, salary, how_to_apply):
    cur = mysql.connection.cursor()
    cur.execute('''INSERT INTO JobOfferList (job_title, company_name, about_company, company_address, job_description, skills, skills_categories, key_responsibility, qualification, benefits, salary, how_to_apply) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                   (job_title, company_name, about_company, company_address, job_description, ', '.join(skills), ', '.join(skill_categories), key_responsibility, qualification, benefits, salary, how_to_apply))
    mysql.connection.commit()
    cur.close()

def create_job_offer_table():
    cur = mysql.connection.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS JobOfferList (
                   id INT AUTO_INCREMENT PRIMARY KEY,
                   job_title VARCHAR(255),
                   company_name VARCHAR(255),
                   about_company TEXT,
                   company_address TEXT,
                   job_description TEXT,
                   key_responsibility TEXT,
                   qualification TEXT,
                   benefits TEXT,
                   salary VARCHAR(100),
                   how_to_apply TEXT,
                   skills TEXT)''') 
    mysql.connection.commit()
    cur.close()


