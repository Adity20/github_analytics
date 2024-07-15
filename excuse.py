import streamlit as st
import requests
import pandas as pd
from datetime import datetime

def fetch_user_repositories(username):
    url = f'https://api.github.com/users/{username}/repos'
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        return repos
    else:
        st.error('Failed to fetch repositories. Please check the username and try again.')
        return []

def fetch_user_overview(username):
    url = f'https://api.github.com/users/{username}'
    response = requests.get(url)
    if response.status_code == 200:
        user = response.json()
        return user
    else:
        st.error('Failed to fetch user data. Please check the username and try again.')
        return None

def generate_readme_content(user, repos, selected_languages, current_project, learning, bio):
    languages = ", ".join(selected_languages)
    repo_names = [repo['name'] for repo in repos]
    commit_history = [repo['pushed_at'] for repo in repos]
    commit_dates = pd.to_datetime(commit_history).date
    commit_counts = pd.Series(commit_dates).value_counts().sort_index()
    
    readme_content = f"""
    <h1 align="center">Hi 👋, I'm {user['name']}</h1>
    <h3 align="center">A passionate developer from {user.get('location', 'somewhere')}</h3>

    <p align="left"> <img src="https://komarev.com/ghpvc/?username={user['login']}&label=Profile%20views&color=0e75b6&style=flat" alt="{user['login']}" /> </p>

    - 🌱 I’m currently learning **{learning}**

    - 💬 Ask me about **{languages}**

    - 📫 How to reach me **{user['email'] if user['email'] else 'N/A'}**

    - 📄 Know about my experiences [Resume](#)

    <h3 align="left">Languages and Tools:</h3>
    <p align="left">
    {"".join([f'<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/{lang.lower()}/{lang.lower()}-original.svg" alt="{lang}" width="40" height="40"/> ' for lang in selected_languages])}
    </p>

    <h3 align="left">📈 GitHub Stats:</h3>
    <p align="left">
    <img align="center" src="https://github-readme-stats.vercel.app/api?username={user['login']}&show_icons=true&locale=en" alt="{user['login']}" />
    <img align="center" src="https://github-readme-streak-stats.herokuapp.com/?user={user['login']}&" alt="{user['login']}" />
    </p>

    <h3 align="left">Projects:</h3>
    <ul>
    {"".join([f'<li><a href="https://github.com/{{user[\'login\']}}/{repo}">{repo}</a></li>' for repo in repo_names])}
    </ul>

    <h3 align="left">📊 Commit History:</h3>
    <p align="left">
    {commit_counts.to_html()}
    </p>
    """
    return readme_content

def main():
    st.title('GitHub Profile README Generator')

    st.markdown("""
        This tool helps you create a beautiful README file for your GitHub profile. 
        Fill in the details below and generate your custom README file.
    """)

    username = st.text_input('Enter GitHub Username', help="GitHub username of the account you want to analyze.")
    if username:
        user = fetch_user_overview(username)
        repos = fetch_user_repositories(username)
        if user and repos:
            bio = st.text_area("Bio", "A passionate developer from somewhere.")
            selected_languages = st.multiselect("Select Languages", 
                                                ['Python', 'JavaScript', 'Java', 'C++', 'Ruby', 'Go', 'Rust'])
            current_project = st.text_input("Current Project", "Working on cool projects.")
            learning = st.text_input("Currently Learning", "Something interesting.")

            if st.button("Generate README"):
                readme_content = generate_readme_content(user, repos, selected_languages, current_project, learning, bio)
                st.markdown(readme_content, unsafe_allow_html=True)
                st.download_button("Download README", readme_content, file_name="README.md", mime="text/markdown")

if __name__ == '__main__':
    main()
