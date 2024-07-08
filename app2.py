import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import Counter
import mindsdb_sdk
import re
import sqlite3

# Initialize NLP pipelines
sentiment_analyzer = pipeline('sentiment-analysis')
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Initialize MindsDB
server = mindsdb_sdk.connect()

# Function to fetch user repositories
def fetch_user_repositories(username):
    url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        repo_names = [repo['name'] for repo in repos]
        return repo_names
    else:
        st.error('Failed to fetch repositories. Please check the username and try again.')
        return []

# Function to fetch repository data
def fetch_repository_data(username, repo_name):
    url = f'https://api.github.com/repos/{username}/{repo_name}'
    response = requests.get(url)
    if response.status_code == 200:
        repo = response.json()
        repo_data = {
            'Name': repo['name'],
            'Description': repo['description'],
            'Stars': repo['stargazers_count'],
            'Forks': repo['forks_count'],
            'Watchers': repo['watchers_count']
        }
        return repo_data
    else:
        st.error('Failed to fetch repository data. Please check the repository name and try again.')
        return None

# Function to fetch commit messages
def fetch_commit_messages(username, repo_name):
    url = f'https://api.github.com/repos/{username}/{repo_name}/commits'
    response = requests.get(url)
    if response.status_code == 200:
        commits = response.json()
        messages = [commit['commit']['message'] for commit in commits]
        return messages
    else:
        st.error('Failed to fetch commit messages.')
        return []

# Function to fetch issues
def fetch_issues(username, repo_name):
    url = f'https://api.github.com/repos/{username}/{repo_name}/issues'
    response = requests.get(url)
    if response.status_code == 200:
        issues = response.json()
        return issues
    else:
        st.error('Failed to fetch issues.')
        return []

# Function to fetch comments on an issue
def fetch_issue_comments(username, repo_name, issue_number):
    url = f'https://api.github.com/repos/{username}/{repo_name}/issues/{issue_number}/comments'
    response = requests.get(url)
    if response.status_code == 200:
        comments = response.json()
        return [comment['body'] for comment in comments]
    else:
        st.error(f'Failed to fetch comments for issue #{issue_number}.')
        return []

# Function to fetch user overview data
def fetch_user_overview(username):
    url = f'https://api.github.com/users/{username}'
    response = requests.get(url)
    if response.status_code == 200:
        user = response.json()
        overview_data = {
            'Username': user['login'],
            'Name': user.get('name', 'N/A'),
            'Public Repos': user['public_repos'],
            'Followers': user['followers'],
            'Following': user['following']
        }
        return overview_data
    else:
        st.error('Failed to fetch user data. Please check the username and try again.')
        return None

# Function to perform keyword extraction
def extract_keywords(texts):
    keywords = []
    for text in texts:
        ner_results = ner(text)
        keywords.append(ner_results)
    return keywords

# Function to perform sentiment analysis
def analyze_sentiment(texts):
    sentiments = sentiment_analyzer(texts)
    return sentiments

# Function to analyze issues
def analyze_issues(issues):
    open_issues = [issue for issue in issues if issue['state'] == 'open']
    closed_issues = [issue for issue in issues if issue['state'] == 'closed']
    labels = [label['name'] for issue in issues for label in issue['labels']]
    return len(open_issues), len(closed_issues), labels

# Function to predict issue complexity or priority
def predict_issue_priority(issue_descriptions):
    # Example simple heuristic for priority, replace with MindsDB prediction
    predictions = ["High" if "urgent" in desc.lower() or "critical" in desc.lower() else "Low" for desc in issue_descriptions]
    return predictions

# Function to identify tech stack demanded
def identify_tech_stack(texts):
    tech_keywords = ["python", "javascript", "react", "docker", "kubernetes", "aws", "azure", "tensorflow", "pytorch"]
    tech_stack = []
    for text in texts:
        found_tech = [tech for tech in tech_keywords if tech in text.lower()]
        tech_stack.extend(found_tech)
    return Counter(tech_stack)

# Function to provide recommendations for issues
def provide_recommendations(issues):
    recommendations = []
    for issue in issues:
        issue_content = issue['title'] + ". " + issue.get('body', '')
        answer = qa_pipeline({
            'context': issue_content,
            'question': "How can this issue be resolved?"
        })
        recommendations.append(answer['answer'])
    return recommendations

# Function to create MindsDB project and train predictor
def create_mindsdb_project_and_train(username, repo_name):
    conn = sqlite3.connect("github_data.db")
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS issues (
        id INTEGER PRIMARY KEY,
        title TEXT,
        description TEXT,
        solution TEXT
    )
''')
    
    # Assuming you have a list of issue data from GitHub API
    issues_data = fetch_issues(username, repo_name)
    
    # Insert issue data into SQLite database
    for issue in issues_data:
        cursor.executemany('''
    INSERT OR IGNORE INTO issues (id, title, description, solution)
    VALUES (?, ?, ?, ?)
''', issue)
    
    conn.commit()

    # Create MindsDB project and train predictor
    server.ml_engines.create(
        "minds_endpoint_engine",
        "minds_endpoint",
        connection_data={"minds_endpoint_api_key": '87c09b70b59d7f096e3332f8dfabea230a343556e3ee59d2000ed49f36d4faf0'},
    )
    
    project = server.create_project(name='github_analytics', from_data="SELECT * FROM issues")
    project = server.get_project('github_analytics')
    project.models.create(
        name="issue_priority",
        predict="solution",
        engine="minds_endpoint_engine",
        max_tokens=512,
        prompt_template="Generate the solution for the issue",
    )
    
    github_data = project.models.get('issue_priority')
    
    # Fetch issue descriptions
    issue_descriptions = pd.read_sql_query("SELECT id, title || '. ' || body AS description FROM issues", conn)
    
    # Make predictions
    predictions = github_data.predict(predictor_name='issue_priority', data=issue_descriptions['description'].tolist())
    
    # Add predictions to issue_descriptions DataFrame
    issue_descriptions['predicted_state'] = predictions
    
    # Close the database connection
    conn.close()
    
    # Return the DataFrame with predictions
    return issue_descriptions

# Streamlit app
st.set_page_config(page_title='GitHub Analytics Dashboard', layout='wide')

st.title('🌟 GitHub Analytics Dashboard')
st.markdown("""
    Welcome to the GitHub Analytics Dashboard! 
    Enter a GitHub username to get started and view detailed analytics about the user's repositories and activity.
""")

# Input fields for GitHub username and repository
with st.sidebar:
    st.header('Input Parameters')
    username = st.text_input('Enter GitHub Username', help="GitHub username of the account you want to analyze.")
    repo_name = st.text_input('Enter Repository Name (optional)', help="Specific repository name to get detailed insights.")

if username:
    if repo_name:
        # Fetch and display repository data
        repo_data = fetch_repository_data(username, repo_name)
        if repo_data:
            st.subheader(f'Repository Information: {repo_name}')
            repo_df = pd.DataFrame([repo_data])
            st.dataframe(repo_df)

            # Visualize stars, forks, and watchers
            st.subheader('Repository Metrics')
            metrics_df = pd.DataFrame(repo_data.items(), columns=['Metric', 'Count']).iloc[2:]
            st.bar_chart(metrics_df.set_index('Metric'))

            # Fetch and analyze commit messages
            commit_messages = fetch_commit_messages(username, repo_name)
            if commit_messages:
                st.subheader('Commit Message Sentiment Analysis')
                sentiments = analyze_sentiment(commit_messages)
                sentiment_df = pd.DataFrame(sentiments)
                st.dataframe(sentiment_df)

                # Sentiment visualization
                st.subheader('Sentiment Distribution')
                sentiment_fig = px.histogram(sentiment_df, x='label', title='Commit Message Sentiment Distribution')
                st.plotly_chart(sentiment_fig)

            # Fetch and analyze issues
            issues = fetch_issues(username, repo_name)
            if issues:
                open_issues_count, closed_issues_count, issue_labels = analyze_issues(issues)
                st.subheader('Issue Analysis')
                st.write(f'Open Issues: {open_issues_count}')
                st.write(f'Closed Issues: {closed_issues_count}')

                # Issue labels visualization
                labels_count = Counter(issue_labels)
                labels_df = pd.DataFrame(labels_count.items(), columns=['Label', 'Count'])
                st.subheader('Issue Labels Distribution')
                labels_fig = px.bar(labels_df, x='Label', y='Count', title='Issue Labels Distribution')
                st.plotly_chart(labels_fig)

                # Issue priority prediction
                st.subheader('Issue Priority Prediction')
                issue_descriptions = [issue['title'] + ". " + issue.get('body', '') for issue in issues]
                issue_priorities = predict_issue_priority(issue_descriptions)
                issue_priority_df = pd.DataFrame({'Issue': issue_descriptions, 'Priority': issue_priorities})
                st.dataframe(issue_priority_df)

                # Recommendations
                st.subheader('Issue Recommendations')
                recommendations = provide_recommendations(issues)
                for idx, recommendation in enumerate(recommendations):
                    st.write(f"Issue #{issues[idx]['number']}: {recommendation}")
    else:
        # Fetch and display user overview data
        user_data = fetch_user_overview(username)
        if user_data:
            st.subheader(f'User Overview: {username}')
            user_df = pd.DataFrame([user_data])
            st.dataframe(user_df)

            # Fetch and analyze user's repositories
            st.subheader('User Repositories')
            repo_names = fetch_user_repositories(username)
            if repo_names:
                st.write(f'Total Repositories: {len(repo_names)}')
                st.write('Repositories:', ', '.join(repo_names))

            # Fetch and visualize user activity (commit history)
            commit_counts = []
            for repo in repo_names:
                commit_messages = fetch_commit_messages(username, repo)
                commit_counts.append(len(commit_messages))

            st.subheader('Commit History')
            commit_history_df = pd.DataFrame({'Repository': repo_names, 'Commits': commit_counts})
            commit_history_fig = px.bar(commit_history_df, x='Repository', y='Commits', title='Commit History by Repository')
            st.plotly_chart(commit_history_fig)

             # Identify commonly used tech stack with error handling
            def fetch_description_safe(username, repo):
                repo_data = fetch_repository_data(username, repo)
                return repo_data.get('description', '') if repo_data else ''

            descriptions = [fetch_description_safe(username, repo) for repo in repo_names if fetch_description_safe(username, repo)]
            tech_stack = identify_tech_stack(descriptions)
            st.subheader('Commonly Used Tech Stack')
            tech_stack_df = pd.DataFrame(tech_stack.items(), columns=['Technology', 'Count'])
            tech_stack_fig = px.bar(tech_stack_df, x='Technology', y='Count', title='Commonly Used Tech Stack')
            st.plotly_chart(tech_stack_fig)