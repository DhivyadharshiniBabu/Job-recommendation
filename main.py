from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import io
from matplotlib.figure import Figure
from wordcloud import WordCloud
from collections import Counter

app = Flask(__name__)

# Load the jobs data from the CSV file
df = pd.read_csv('Jobs_2.csv')

# Load the available skills from PrioritySkills.csv
available_skills_df = pd.read_csv('PrioritySkills.csv')
available_skills = available_skills_df['word'].tolist()

dfx = pd.read_csv('PrioritySkills.csv')

# Define the model
tfidf_vectorizer = TfidfVectorizer()
job_description_vectors = tfidf_vectorizer.fit_transform(df['description'])
model = NearestNeighbors(n_neighbors=10, metric='cosine')
model.fit(job_description_vectors)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_skills = request.form.get('user_skills')
        user_description = request.form.get('user_description')

        if user_skills is None:
            user_skills = ""  # Set it to an empty string if it's None

        # Check if any user skills match with the dataset
        matching_jobs = df[df['tokens_filtered'].apply(lambda x: any(skill in x for skill in user_skills.split()))]

        if matching_jobs.empty:
            return render_template('home.html', user_input='', results="No suitable jobs found", available_skills=available_skills)
        else:
            # Combine user skills and user description
            user_input = user_skills + " " + user_description

            # Vectorize user input
            user_vector = tfidf_vectorizer.transform([user_input])

            # Find the nearest neighbors for the user input
            user_neighbors = model.kneighbors(user_vector, return_distance=False)

            if len(user_neighbors) == 0:
                results = "No suitable jobs found"
            else:
                # Get the top 10 similar jobs
                similar_jobs = df.iloc[user_neighbors[0]][['title', 'company', 'location']]
                results = similar_jobs.to_dict(orient='records')

            return render_template('home.html', user_input=user_input, results=results, available_skills=available_skills)

    return render_template('home.html', user_input='', results=None, available_skills=available_skills)

@app.route('/hired/<company>/<title>/<location>')
def hired(company, title, location):
    return render_template('hired.html', company=company, title=title, location=location)

def visual():
    # Load the data from Jobs_2.csv and PrioritySkills.csv
    jobs_df = pd.read_csv('Jobs_2.csv')
    skills_df = pd.read_csv('PrioritySkills.csv')

    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Visualization 1: Top 10 Skills Counts
    top_skills = skills_df.head(10)
    ax.bar(top_skills['word'], top_skills['count'])
    ax.set_title('Top 10 Skills Counts')
    ax.tick_params(axis='x', rotation=45)

    # Calculate the skill counts
    skill_counts = skills_df['count']
    total_skills = skill_counts.sum()

    # Calculate the percentage of each skill
    skill_percentages = (skill_counts / total_skills) * 100

    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Visualization 2: Skill Percentage Distribution
    ax.pie(skill_percentages, labels=skills_df['word'], autopct='%1.1f%%', startangle=90)
    ax.set_title('Skill Percentage Distribution')

    # Visualization 3: Word Cloud for Job Descriptions
    job_descriptions = " ".join(jobs_df['description'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(job_descriptions)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Job Descriptions')


    # Visualization 4: Top 10 Most Frequent Words in Job Descriptions
    job_descriptions = " ".join(jobs_df['description'])
    words = job_descriptions.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(10)
    words, counts = zip(*top_words)

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=words, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Top 10 Most Frequent Words in Job Descriptions')


    # Visualization 5: Top 10 Job Locations
    top_locations = jobs_df['location'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    top_locations.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Job Locations')
    plt.xlabel('Location')
    plt.ylabel('Number of Jobs')
    plt.xticks(rotation=45)


    # Visualization 6: Top 10 Companies
    top_companies = jobs_df['company'].value_counts().head(10)

    plt.figure(figsize=(8, 8))
    plt.pie(top_companies, labels=top_companies.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85, labeldistance=1.05)
    plt.title('Top 10 Companies')
    plt.axis('equal')
    plt.legend(loc='upper left', bbox_to_anchor=(0.85, 1))


    # Visualization 7: Top 10 Skills (Bar Chart)
    top_skills = skills_df.sort_values(by='count', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(top_skills['word'], top_skills['count'])
    plt.title('Top 10 Skills')
    plt.xlabel('Skills')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Visualization 8: Scatter Plot (Job Title Length vs. Number of Skills)
    jobs_df['title_length'] = jobs_df['title'].apply(len)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(jobs_df['title_length'], jobs_df['tokens_filtered'].apply(len), alpha=0.5)
    plt.title('Scatter Plot: Job Title Length vs. Number of Skills')
    plt.xlabel('Job Title Length')
    plt.ylabel('Number of Skills')
    plt.grid(True)
    
@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)
