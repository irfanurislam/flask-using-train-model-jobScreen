import MySQLdb
import string

# Connect to MySQL
def get_mysql_connection(config):
    mysql = MySQLdb.connect(host=config['MYSQL_HOST'], user=config['MYSQL_USER'], 
                            password=config['MYSQL_PASSWORD'], db=config['MYSQL_DB'])
    return mysql

def word_count(mysql):
    cursor = mysql.cursor()
    query = "SELECT skills, job_title FROM jobofferlist"
    cursor.execute(query)
    data = cursor.fetchall()
    
    # Initialize word count dictionaries for each column
    skills_word_count = {}
    job_title_word_count = {}

    # Define punctuation characters to remove
    punctuation_chars = set(string.punctuation)

    for row in data:
        # Process skills column
        skills_text = row[0]
        # Remove punctuation
        skills_text_cleaned = ''.join(ch for ch in skills_text if ch not in punctuation_chars)
        skills_words = skills_text_cleaned.split()
        for word in skills_words:
            skills_word_count[word] = skills_word_count.get(word, 0) + 1

        # Process job title column
        job_title_text = row[1]
        # Remove punctuation
        job_title_text_cleaned = ''.join(ch for ch in job_title_text if ch not in punctuation_chars)
        job_title_words = job_title_text_cleaned.split()
        for word in job_title_words:
            job_title_word_count[word] = job_title_word_count.get(word, 0) + 1

    # Sort the word count dictionaries by word count in descending order
    sorted_skills_word_count = dict(sorted(skills_word_count.items(), key=lambda item: item[1], reverse=True))
    sorted_job_title_word_count = dict(sorted(job_title_word_count.items(), key=lambda item: item[1], reverse=True))

    # Retrieve the first 15 words with the most occurrences for each column
    top_15_skills_words = {k: sorted_skills_word_count[k] for k in list(sorted_skills_word_count)[:15]}
    top_15_job_title_words = {k: sorted_job_title_word_count[k] for k in list(sorted_job_title_word_count)[:15]}
    
    # Generate serial numbers manually
    serial_numbers = list(range(1, 16))

    # Zip the serial numbers with the top 15 words for each column
    top_15_skills_words_with_serial = [(serial_numbers[i], word, count) for i, (word, count) in enumerate(top_15_skills_words.items())]
    top_15_job_title_words_with_serial = [(serial_numbers[i], word, count) for i, (word, count) in enumerate(top_15_job_title_words.items())]

    return top_15_skills_words_with_serial, top_15_job_title_words_with_serial
