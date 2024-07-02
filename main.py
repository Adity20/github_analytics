import streamlit as st
import requests
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from collections import Counter
import mindsdb_sdk
import re
import sqlite3
import torch


# Initializing the  NLP pipelines
sentiment_analyzer = pipeline('sentiment-analysis',model='model="distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# connect to the mindsdb_sdk
server = mindsdb_sdk.connect()

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

def fetch_issues(username, repo_name):
    url = f'https://api.github.com/repos/{username}/{repo_name}/issues'
    response = requests.get(url)
    if response.status_code == 200:
        issues = response.json()
        return issues
    else:
        st.error('Failed to fetch issues.')
        return []

def fetch_issue_comments(username, repo_name, issue_number):
    url = f'https://api.github.com/repos/{username}/{repo_name}/issues/{issue_number}/comments'
    response = requests.get(url)
    if response.status_code == 200:
        comments = response.json()
        return [comment['body'] for comment in comments]
    else:
        st.error(f'Failed to fetch comments for issue #{issue_number}.')
        return []

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

def extract_keywords(texts):
    keywords = []
    for text in texts:
        ner_results = ner(text)
        keywords.append(ner_results)
    return keywords

def analyze_sentiment(texts):
    sentiments = sentiment_analyzer(texts)
    return sentiments

def analyze_issues(issues):
    open_issues = [issue for issue in issues if issue['state'] == 'open']
    closed_issues = [issue for issue in issues if issue['state'] == 'closed']
    labels = [label['name'] for issue in issues for label in issue['labels']]
    return len(open_issues), len(closed_issues), labels


def predict_issue_priority(issue_descriptions):
    # Example simple heuristic for priority, replace with MindsDB prediction
    predictions = ["High" if "urgent" in desc.lower() or "critical" in desc.lower() else "Low" for desc in issue_descriptions]
    return predictions

def identify_tech_stack(texts):
    tech_keywords = ["python", "javascript", "react", "docker", "kubernetes", "aws", "azure", "tensorflow", "pytorch"]
    tech_stack = []
    for text in texts:
        found_tech = [tech for tech in tech_keywords if tech in text.lower()]
        tech_stack.extend(found_tech)
    return Counter(tech_stack)


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

def create_mindsdb_project_and_train(username, repo_name):
    conn = sqlite3.connect("github_data.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS issues (
                      id INTEGER PRIMARY KEY, 
                      title TEXT, 
                      body TEXT, 
                      state TEXT, 
                      created_at TEXT, 
                      updated_at TEXT)''')

    issues_data = fetch_issues(username, repo_name)
    
    # Insert issue data into SQLite database
    for issue in issues_data:
        cursor.execute('''INSERT OR IGNORE INTO issues (id, title, body, state, created_at, updated_at)
                          VALUES (?, ?, ?, ?, ?, ?)''', 
                          (issue['id'], issue['title'], issue.get('body', ''), issue['state'], 
                           issue['created_at'], issue['updated_at']))
    
    conn.commit()
    conn.close()
    
    server.ml_engines.create(
        "minds_endpoint_engine",
        "minds_endpoint",
        connection_data={"minds_endpoint_api_key": 'API_KEY_HERE'},
    )
    project = server.create_project(name='github_analytics', from_data="SELECT * FROM issues")
    project = server.get_project('github_analytics')
    project.models.create(
        name="issue_priority",
        predict="state",
        engine="minds_endpoint_engine",
        max_tokens=512,
        prompt_template="Generate the state",
    )
    github_data = project.models.get('issue_priority')
    github_data.predict('state')


st.set_page_config(page_title='GitHub Analytics Dashboard', layout='wide')

st.title('🌟 GitHub Analytics Dashboard')
st.markdown("""
    Welcome to the GitHub Analytics Dashboard! 
    Enter a GitHub username to get started and view detailed analytics about the user's repositories and activity.
""")

with st.sidebar:
    st.header('Input Parameters')
    username = st.text_input('Enter GitHub Username', help="GitHub username of the account you want to analyze.")
    repo_name = st.text_input('Enter Repository Name (optional)', help="Specific repository name to get detailed insights.")

if username:
    if repo_name:
        repo_data = fetch_repository_data(username, repo_name)
        if repo_data:
            st.subheader(f'Repository Information: {repo_name}')
            repo_df = pd.DataFrame([repo_data])
            st.dataframe(repo_df)
            st.subheader('Repository Metrics')
            metrics_df = pd.DataFrame(repo_data.items(), columns=['Metric', 'Count']).iloc[2:]
            st.bar_chart(metrics_df.set_index('Metric'))

            commit_messages = fetch_commit_messages(username, repo_name)
            if commit_messages:
                st.subheader('Commit Messages Analysis')

                st.write(commit_messages)

                st.subheader('Keywords in Commit Messages')
                keywords = extract_keywords(commit_messages)
                keywords_df = pd.DataFrame([{
                    'Commit': commit_messages[i],
                    'Keywords': [keyword['word'] for keyword in keywords[i]]
                } for i in range(len(commit_messages))])
                st.dataframe(keywords_df)

                st.subheader('Sentiment Analysis of Commit Messages')
                sentiments = analyze_sentiment(commit_messages)
                sentiment_df = pd.DataFrame(sentiments)
                st.dataframe(sentiment_df)

            issues = fetch_issues(username, repo_name)
            if issues:
                st.subheader('Issue Reporting Trends')
                open_issues, closed_issues, labels = analyze_issues(issues)
                st.write(f"Open Issues: {open_issues}")
                st.write(f"Closed Issues: {closed_issues}")
                st.write(f"Issue Labels: {', '.join(set(labels))}")

                issue_descriptions = [issue['title'] + ". " + issue.get('body', '') for issue in issues]
                priority_predictions = predict_issue_priority(issue_descriptions)
                priority_df = pd.DataFrame({
                    'Issue': [issue['title'] for issue in issues],
                    'Priority': priority_predictions
                })
                st.dataframe(priority_df)

                st.subheader('Comments Analysis')
                comments = [fetch_issue_comments(username, repo_name, issue['number']) for issue in issues]
                flat_comments = [comment for sublist in comments for comment in sublist]
                comment_sentiments = analyze_sentiment(flat_comments)
                comment_sentiments_df = pd.DataFrame(comment_sentiments)
                st.dataframe(comment_sentiments_df)
                st.subheader('Tech Stack Demanded')
                tech_stack = identify_tech_stack(issue_descriptions + flat_comments)
                tech_stack_df = pd.DataFrame(tech_stack.items(), columns=['Technology', 'Count'])
                st.dataframe(tech_stack_df)
                st.subheader('Recommendations for Issues')
                recommendations = provide_recommendations(issues)
                recommendations_df = pd.DataFrame({
                    'Issue': [issue['title'] for issue in issues],
                    'Recommendation': recommendations
                })
                st.dataframe(recommendations_df)

                # # MindsDB integration for predictive analytics
                # st.subheader('MindsDB Predictions')
                # create_mindsdb_project_and_train(username,repo_name)
                # # Assuming you have a trained model, you can use it to make predictions
                # conn = sqlite3.connect("github_data.db")
                # project = server.get_project('github_analytics')
                # project.create_model('issue_priority', model_type='predictor')
                # github_data = project.models.get('issue_priority')
                # issue_descriptions = pd.read_sql_query("SELECT title || '. ' || body AS description FROM issues", conn)
                # predictions = github_data.predict(predictor_name='issue_priority', data=issue_descriptions)
                # predictions_df = pd.DataFrame(predictions)
                # st.dataframe(predictions_df)
    else:
        # Fetch and display user overview data
        overview_data = fetch_user_overview(username)
        if overview_data:
            st.subheader('User Overview')
            overview_df = pd.DataFrame([overview_data])
            st.dataframe(overview_df)
            
            # Fetch and display user repositories
            st.subheader('User Repositories')
            repo_names = fetch_user_repositories(username)
            if repo_names:
                st.write(', '.join(repo_names))
else:
    st.info('Please enter a GitHub username to get started.')
