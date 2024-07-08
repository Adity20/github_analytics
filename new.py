import sqlite3
import mindsdb_sdk
import pandas as pd
import requests
import streamlit as st

# Initialize MindsDB
server = mindsdb_sdk.connect()

def fetch_issues(username, repo_name):
    url = f'https://api.github.com/repos/{username}/{repo_name}/issues'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error('Failed to fetch issues.')
        return []

# Function to create MindsDB project and train predictor
def create_mindsdb_project_and_train(username, repo_name):
    conn = sqlite3.connect("githubdata.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS issues (
                      id INTEGER PRIMARY KEY, 
                      title TEXT, 
                      body TEXT, 
                      state TEXT, 
                      created_at TEXT, 
                      updated_at TEXT)''')
    
    issues_data = fetch_issues(username, repo_name)
    for issue in issues_data:
        cursor.execute('''INSERT OR IGNORE INTO issues (id, title, body, state, created_at, updated_at)
                          VALUES (?, ?, ?, ?, ?, ?)''', 
                          (issue['id'], issue['title'], issue.get('body', ''), issue['state'], 
                           issue['created_at'], issue['updated_at']))
    
    conn.commit()
    conn.close()
    
    # Check if MindsDB engine already exists
    engine_name = "minds_endpoint_engine"
    engines = server.ml_engines.list()
    engine_exists = any(engine['name'] == engine_name for engine in engines)
    
    if not engine_exists:
        server.ml_engines.create(
            engine_name,
            "minds_endpoint",
            connection_data={"minds_endpoint_api_key": 'key'},
        )
    
    project_name = 'issue_solution_predictor'
    # Check if project already exists
    projects = server.projects.list()
    project_exists = any(project['name'] == project_name for project in projects)
    
    if not project_exists:
        project = server.create_project(name=project_name)
    else:
        project = server.get_project(project_name)
    
    model_name = 'issue_solution_predictor'
    # Check if model already exists
    models = project.models.list()
    model_exists = any(model['name'] == model_name for model in models)
    
    if not model_exists:
        project.models.create(
            name=model_name,
            predict="solution",
            engine=engine_name,
            max_tokens=512,
            prompt_template="Given the issue description, generate a possible solution",
        )
    
    github_data = project.models.get(model_name)
    return github_data

# Streamlit App
st.set_page_config(page_title='Issue Solution Predictor', layout='wide')
st.title('🔍 Issue Solution Predictor using MindsDB and Google Gemini')

username = st.text_input('Enter GitHub Username')
repo_name = st.text_input('Enter Repository Name')

if st.button('Fetch and Predict'):
    if username and repo_name:
        st.write('Fetching issues and training model...')
        github_data = create_mindsdb_project_and_train(username, repo_name)
        
        conn = sqlite3.connect("githubdata.db")
        issue_descriptions = pd.read_sql_query("SELECT title || '. ' || body AS description FROM issues", conn)
        predictions = github_data.predict(predictor_name='issue_solution_predictor', data=issue_descriptions)
        predictions_df = pd.DataFrame(predictions)
        st.write('Predictions:', predictions_df)
    else:
        st.error('Please enter both username and repository name.')
