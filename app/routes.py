from app import app, mysql, models
from flask import render_template, request, session, redirect, url_for
import mysql.connector
from mysql.connector import Error
from app.models import create_job_offer_table, insert_job_list
from werkzeug.security import generate_password_hash, check_password_hash
# app = Flask(__name__)
#Database Connnection
app.secret_key = 'your_secret_key'
db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="12345",
            database="jobofferlist"
        )

# Creating cursor object
cursor = db.cursor(dictionary=True)

# Create users table if not exists
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, username VARCHAR(255) UNIQUE, password VARCHAR(255))")

#Main Route
# @app.route('/')
# def index():
#    if 'username' in session:      
#     return render_template('index.html', username=session['username'])
#    return redirect(url_for('login')) 
@app.route('/')
def index():
    if 'username' in session:
        logged_in = True
        username = session['username']
    else:
        logged_in = False
        username = None
    return render_template('index.html', logged_in=logged_in, username=username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            db.commit()
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            return render_template('register.html', error='Username already exists')
    return render_template('register.html')

@app.route('/userlist')
def user_list():
    if 'username' in session:
        return render_template('userlist.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

#  newly added upon
@app.route('/create_table')
def create_table():
    create_job_offer_table()
    return 'JobOfferList table created successfully!'

# Route to delete a job offer
@app.route('/delete_job_offer/<int:job_offer_id>', methods=['GET'])
def delete_job_offer(job_offer_id):
    cur = db.cursor()
    cur.execute("DELETE FROM JobOfferList WHERE id = %s", (job_offer_id,))
    db.commit()
    cur.close()
    return redirect(url_for('get_job_offers'))

# Route to edit a job offer
@app.route('/edit_job_offer/<int:job_offer_id>', methods=['GET', 'POST'])
def edit_job_offer(job_offer_id):
    if request.method == 'GET':
        cur = db.cursor()
        cur.execute("SELECT * FROM JobOfferList WHERE id = %s", (job_offer_id,))
        job_offer = cur.fetchone()
        cur.close()
        return render_template('edit_job_offer.html', job_offer=job_offer)
    elif request.method == 'POST':
        # Update the job offer in the database based on the form data
        job_title = request.form['job_title']
        company_name = request.form['company_name']
        skills = request.form['skills']
        # Update other fields similarly
        cur = db.cursor()
        cur.execute("UPDATE JobOfferList SET job_title = %s, company_name = %s, skills = %s WHERE id = %s",
                    (job_title, company_name, skills, job_offer_id))
        db.commit()
        cur.close()
        return redirect(url_for('job_offer_list'))

## Route to view a job offer
@app.route('/view_job_offer/<int:job_offer_id>', methods=['GET'])
def view_job_offer(job_offer_id):
    cur = db.cursor()
    cur.execute("SELECT * FROM JobOfferList WHERE id = %s", (job_offer_id,))
    job_offer = cur.fetchone()
    cur.close()
    return render_template('view_job_offer.html', job_offer=job_offer)

#Job Offer route
@app.route('/job_list', methods=['GET'])
def job_offer_list():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page
    
    # Retrieve job offers for the current page
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM jobofferlist LIMIT %s OFFSET %s", (per_page, offset))
    job_offers = cursor.fetchall()
    
    # Determine the total number of job offers
    cursor.execute("SELECT COUNT(*) AS total FROM jobofferlist")
    total_jobs = cursor.fetchone()['total']

    return render_template('job_list.html', job_offers=job_offers, total_jobs=total_jobs, per_page=per_page)

@app.route('/job/<int:job_id>')
def job_details(job_id):
    # Retrieve job details from the database based on job_id
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM jobofferlist WHERE id = %s", (job_id,))
    job = cursor.fetchone()
    return render_template('job_details.html', job=job, job_id=job_id)

# Route to display the upload form
@app.route('/upload/<int:job_id>')
def upload_form(job_id):
    # Retrieve job details from the database
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT job_title, company_name FROM jobofferlist WHERE id = %s", (job_id,))
    job = cursor.fetchone()
    job_title = job['job_title']
    company_name = job['company_name']
    
    return render_template('upload.html', job_id=job_id, job_title=job_title, company_name=company_name)


from flask import render_template, request, redirect, url_for
import models   
skill_categories = {
        'Web Development': ['HTML', 'CSS', 'JavaScript', 'PHP', 'MySQL'],
        'Programming Languages': ['Java', 'C', 'C++', 'Python'],
        'Frameworks': ['React', 'Angular', 'Django', 'Flask'],
        'Mobile Development': ['Android_Development', 'iOS_Development', 'React_Native', 'Flutter'],
        'Cloud Services': ['AWS', 'Azure', 'Google Cloud'],
        'Data Science': ['Pandas', 'Matplotlib', 'TensorFlow', 'PyTorch'],
        'DevOps': ['Docker', 'Kubernetes', 'Jenkins', 'Ansible'],
        'Operating Systems': ['Unix/Linux', 'Windows', 'macOS'],
        'Networking': ['TCP/IP', 'HTTP', 'DNS', 'Firewalls'],
        'Security': ['Cybersecurity', 'Ethical Hacking', 'Penetration_Testing'],
        'UI/UX Design': ['Adobe_XD', 'Sketch', 'Figma', 'InVision'],
        'Agile Methodologies': ['Scrum', 'Kanban', 'Lean'],
        'Database Systems': ['SQL', 'NoSQL', 'MongoDB', 'PostgreSQL'],
        'Big Data Technologies': ['Hadoop', 'Spark', 'Hive', 'Kafka'],
        'Problem-Solving': ['Critical_Thinking', 'Analytical_Skills', 'Logic'],
        'Communication': ['Verbal_Communication', 'Written_Communication', 'Presentation_Skills'],
        'Teamwork': ['Collaboration_Skills', 'Interpersonal_Skills', 'Team_Building'],
        'Continuous Learning': ['Self-Learning', 'Professional_Development', 'Adaptability'],
    }

@app.route('/insert_job_offer', methods=['GET', 'POST'])
def insert_job_offer():
    if request.method == 'POST':
        job_title = request.form['job_title']
        company_name = request.form['company_name']
        about_company = request.form['about_company']
        company_address = request.form['company_address']
        job_description = request.form['job_description']
        skills = request.form.getlist('skills')
        skill_categories_selected = [category for category, skills_list in skill_categories.items() if any(skill in skills_list for skill in skills)]
        key_responsibility = request.form['key_responsibility']
        qualification = request.form['qualification']
        benefits = request.form['benefits']
        salary = request.form['salary']
        how_to_apply = request.form['how_to_apply']

        insert_job_list(job_title, company_name, about_company, company_address, job_description, skills, skill_categories_selected, key_responsibility, qualification, benefits, salary, how_to_apply)

        return redirect(url_for('success'))
    else:
        return render_template('insert_job_offer.html', skill_categories=skill_categories)

@app.route('/success')
def success():
    return render_template('success.html')

#Fetch Data, Train Model and Predict with Uploaded CV or Resume

from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from flask_mysqldb import MySQL

# Initialize global model
model = None

def fetch_data():
    try:
        cursor = db.cursor()
        cursor.execute("SELECT job_title, skills FROM jobofferlist")  
        data = cursor.fetchall()
        db.close()
        if not data:
            print("No data fetched from the database.")
            return None
        df = pd.DataFrame(data, columns=['job_title', 'skills'])
        return df
    except Error as e:
        print("Error fetching data from MySQL: There are no new Values!", e)
        return None

def preprocess_data(df):
    X = df['job_title'] + ' ' + df['skills']  
    y = df['job_title']
    return X, y

from werkzeug.utils import secure_filename
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

#Model train route
@app.route('/train_model', methods=['GET'])
def train_model():
    global model
    df = fetch_data()
    if df is None:
        return jsonify({"error": "Failed to fetch data from database"}), 500

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model to a folder
    folder_path = "models"   
    os.makedirs(folder_path, exist_ok=True)   
    model_file_path = os.path.join(folder_path, "model.joblib")
    joblib.dump(model, model_file_path)

    return jsonify({"accuracy": accuracy, "model_path": model_file_path})


from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import PyPDF2
from werkzeug.utils import secure_filename

def score_resume(text, required_skills):
    score = 0
    for skill in required_skills:
        if skill.lower() in text.lower():
            score += 1
    return score

# Define the directory for saving generated HTML files
HTML_FILES_DIRECTORY = "app/templates/ranking"

# Ensure the directory exists
os.makedirs(HTML_FILES_DIRECTORY, exist_ok=True)

# Define the directory for saving uploaded resumes
UPLOAD_FOLDER = "resumes"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the directory exists for uploaded resumes
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def rank_resumes_and_generate_html(job_id):
    try:
        # Retrieve job details from the database
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT job_title, skills FROM jobofferlist WHERE id = %s", (job_id,))
        job = cursor.fetchone()

        if job:
            job_title = job['job_title']
            skills = job['skills']

            # Check if skills is not None
            if skills is not None:
                # Define required skills based on job description
                required_skills = skills.split(', ')  

                # Iterate over company directories
                for company_folder in os.listdir("resumes"):
                    company_folder_path = os.path.join("resumes", company_folder)
                    if os.path.isdir(company_folder_path):
                        # Iterate over job title directories within the company
                        for job_title_folder in os.listdir(company_folder_path):
                            job_title_folder_path = os.path.join(company_folder_path, job_title_folder)
                            if os.path.isdir(job_title_folder_path) and job_title_folder == job_title:
                                print(f"Ranking resumes for company: {company_folder}, job title: {job_title_folder}")
                                resume_scores = {}
                                # Collect resume texts and score them
                                for filename in os.listdir(job_title_folder_path):
                                    if filename.endswith(".pdf"):
                                        resume_path = os.path.join(job_title_folder_path, filename)
                                        print(f"Processing resume: {resume_path}") 
                                        resume_text = extract_text_from_pdf(resume_path)
                                        resume_scores[filename] = score_resume(resume_text, required_skills)

                                # Rank resumes based on scores
                                ranked_resumes = sorted(resume_scores.items(), key=lambda x: x[1], reverse=True)

                                # Creating the HTML file
                                html_file_path = os.path.join(HTML_FILES_DIRECTORY, f"Ranked_Resumes_{company_folder} as {job_title_folder}.html")
                                with open(html_file_path, "w") as html_file:
                                    html_file.write("<html><head><title>Ranked Resumes</title>")
                                    html_file.write("<style>")
                                    html_file.write("table {")
                                    html_file.write("    width: 100%;")
                                    html_file.write("    border-collapse: collapse;")
                                    html_file.write("}")
                                    html_file.write("th, td {")
                                    html_file.write("    border: 1px solid #ddd;")
                                    html_file.write("    padding: 8px;")
                                    html_file.write("    text-align: left;")
                                    html_file.write("}")
                                    html_file.write("th {")
                                    html_file.write("    background-color: #f2f2f2;")
                                    html_file.write("}")
                                    html_file.write("</style>")
                                    html_file.write("</head><body>")
                                    html_file.write(f"<h1>Ranked Resumes for {company_folder} - {job_title_folder}</h1>")
                                    html_file.write("<table>")
                                    html_file.write("<tr><th>Rank</th><th>Resume</th><th>Score</th></tr>")
                                    for rank, (resume, score) in enumerate(ranked_resumes, start=1):
                                        html_file.write(f"<tr><td>{rank}</td><td>{resume}</td><td>{score}</td></tr>")
                                    html_file.write("</table>")
                                    html_file.write("</body></html>")
            else:
                print("Job description is None. Skipping...")
        else:
            print("Job not found in the database.")   

    except Exception as e:
        print(f"An error occurred: {e}")  

@app.route('/display')
def display_generated_files():
    # List all generated HTML files
    generated_files = [file for file in os.listdir(HTML_FILES_DIRECTORY) if file.endswith(".html")]
    # Render the HTML template with the list of generated files
    return render_template('generated_files.html', generated_files=generated_files)

@app.route('/serve/<path:filename>')
def serve_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'templates', 'ranking'), filename)

@app.route('/generated-html')
def generated_html():
    generated_files = os.listdir(os.path.join(app.root_path, 'templates', 'ranking'))
    return render_template('generated_html_files.html', generated_files=generated_files)

# DELETE THAT GENERATED COMPANY NAME AND JOB POSITION
@app.route('/delete_file', methods=['POST'])
def delete_file():
    if request.method == 'POST':
        file_name = request.form['file_name']
        file_path = os.path.join(HTML_FILES_DIRECTORY, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return redirect(url_for('display_generated_files'))
        else:
            return "File not found", 404

# Define a function to fetch data from MySQL
def fetch_data_from_mysql():
    curr = db.cursor()
    curr.execute("SELECT job_title, skills FROM jobofferlist")
    data = curr.fetchall()
    curr.close()
    return data

# Load the trained model
pipeline = joblib.load('models/model.joblib')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Function to calculate similarity
def calculate_similarity(resume_text, job_texts):
    # Transform resume text
    resume_vector = pipeline.named_steps['tfidf'].transform([resume_text])
    # Transform job descriptions                                                       
    job_vectors = pipeline.named_steps['tfidf'].transform(job_texts)
    # Calculate cosine similarity
    similarities = cosine_similarity(resume_vector, job_vectors)
    return similarities[0]

@app.route('/file_upload')
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get form data
        job_id = request.form.get('job_id')
        company_name = request.form.get('company_name')
        job_title = request.form.get('job_title')
        resume_file = request.files.get('resume')

        # Check if all required form fields are provided
        if job_id and company_name and job_title and resume_file:
                # Create directory for company if not exists
                company_directory = os.path.join(app.config['UPLOAD_FOLDER'], company_name)
                os.makedirs(company_directory, exist_ok=True)

                # Create directory for job title within company directory
                job_title_directory = os.path.join(company_directory, job_title)
                os.makedirs(job_title_directory, exist_ok=True)

                # Save resume file
                resume_filename = secure_filename(resume_file.filename)
                resume_path = os.path.join(job_title_directory, resume_filename)
                resume_file.save(resume_path)

                # Insert job details into database
                cursor = db.cursor()
                cursor.execute("INSERT INTO uploaded_resume (company_name, job_title, filename) VALUES (%s, %s, %s)",
                               (company_name, job_title, resume_filename))
                db.commit()

                # Rank resumes and generate HTML files for the specified job
                rank_resumes_and_generate_html(job_id)

                resume_text = extract_text_from_pdf(resume_path)

                # Fetch job descriptions from your dataset 
                data = fetch_data_from_mysql()  
                job_texts, _ = zip(*data)

                # Calculate similarity
                similarities = calculate_similarity(resume_text, job_texts)

                # Find the index of the most similar job description
                most_similar_index = similarities.argmax()
                most_similar_job = data[most_similar_index]

                # Render the template with results
                return render_template('upload_success.html', job_id=job_id, most_similar_job=most_similar_job[0], similarity_score=similarities[most_similar_index])
        else:
            return "Please provide all required form fields."
        
    return "Method not allowed."


@app.route('/single_cv_upload')
def single_cv_upload():
    return render_template('upload_cv.html')
    
import os
from tempfile import NamedTemporaryFile

@app.route('/upload_cv', methods=['POST'])
def cv_upload():
    if 'resume' not in request.files:
        return "No file part"
    resume_file = request.files['resume']
    if resume_file.filename == '':
        return "No selected file"
    if resume_file:
        with NamedTemporaryFile(delete=False) as temp_file:
            resume_file.save(temp_file.name)
            resume_text = extract_text_from_pdf(temp_file.name)
        # Fetch job descriptions from your dataset 
        data = fetch_data_from_mysql()
        job_texts, _ = zip(*data)
        # Calculate similarity
        similarities = calculate_similarity(resume_text, job_texts)
        # Find the index of the most similar job description
        most_similar_index = similarities.argmax()
        most_similar_job = data[most_similar_index]
        # Print the most similar job description and its similarity score
        result = "<h2 align=center>Most similar job description: {}\n </h2>".format(most_similar_job[0])  # Assuming the job title is the first element
        result += "<br><h1 align=center>Similarity score:  {}\n </h1>".format(similarities[most_similar_index])
        os.unlink(temp_file.name)  # Remove the temporary file after closing
        return result


    
    
    
from app.company_basis import train_model, preprocess_text, df, retrieve_similar_jobs
# Step 4: Route to train the model
@app.route('/train', methods=['GET'])
def train():
    try:
        accuracy = train_model()
        return render_template('accuracy.html', model_accuracy=accuracy)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['GET', 'POST'])
def search():
    similar_job_titles = []  
    similar_skills_categories = []  
    similar_skill_counts = []
    similar_skills = []
    vectorizer = joblib.load('models/vectorizer.joblib')
    nb_model = joblib.load('models/nb_model.joblib')  # Load Naive Bayes model
    
    if request.method == 'POST':
        job_title = request.form['job_title']
        job_title = preprocess_text(job_title)
        input_text_tfidf = vectorizer.transform([job_title])
        similar_job_indices = nb_model.predict(input_text_tfidf)  # Use Naive Bayes model
        similar_job_titles = [df.iloc[idx]['job_title'] for idx in similar_job_indices.flatten()]
        similar_skills_categories = [df.iloc[idx]['skills_categories'] for idx in similar_job_indices.flatten()]  
        relevant_skills = [df.iloc[idx]['skills'] for idx in similar_job_indices.flatten()]
        similar_skill_counts = [len(skills.split(', ')) for skills in relevant_skills]
        similar_skills = [skills.split(', ') for skills in relevant_skills]

    else:
        job_title = request.args.get('job_title', '')
        job_title = preprocess_text(job_title)
        input_text_tfidf = vectorizer.transform([job_title])
        similar_job_indices = nb_model.predict(input_text_tfidf)  # Use Naive Bayes model
        similar_job_titles = [df.iloc[idx]['job_title'] for idx in similar_job_indices.flatten()]
        similar_skills_categories = [df.iloc[idx]['skills_categories'] for idx in similar_job_indices.flatten()] 
        relevant_skills = [df.iloc[idx]['skills'] for idx in similar_job_indices.flatten()]
        similar_skill_counts = [len(skills.split(', ')) for skills in relevant_skills]
        similar_skills = [skills.split(', ') for skills in relevant_skills]

    return render_template('search.html', job_title=job_title, similar_job_titles=similar_job_titles,
                           similar_skills_categories=similar_skills_categories,  
                           similar_skill_counts=similar_skill_counts, similar_skills=similar_skills, zip=zip)





#Skills and Job raking and counting 
from app.word_count import get_mysql_connection, word_count

# Configure MySQL connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '12345'
app.config['MYSQL_DB'] = 'jobofferlist'

# Connect to MySQL
mysql = get_mysql_connection(app.config)

@app.route('/top_skills')
def skills_count_route():
    top_15_skills_words_with_serial, _ = word_count(mysql) 
    return render_template('top_skills.html', top_15_skills_words_with_serial=top_15_skills_words_with_serial)

@app.route('/top_jobs')
def jobs_count_route():
    _, top_15_job_title_words_with_serial = word_count(mysql) 
    return render_template('top_jobs.html', top_15_job_title_words_with_serial=top_15_job_title_words_with_serial)



