# GitHub Analytics Dashboard

🌟 Welcome to the GitHub Analytics Dashboard! This project provides detailed analytics about GitHub user repositories and activities using advanced NLP techniques and predictive analytics with MindsDB.

## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Demo

<iframe width="560" height="315" src="https://www.youtube.com/watch?v=HqKJo1kIZoU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Features

- **User Overview**: Displays detailed information about a GitHub user, including their repositories, followers, and following counts.
- **Repository Insights**: Provides metrics such as stars, forks, and watchers for a specific repository.
- **Commit Analysis**: Analyzes commit messages for keyword extraction and sentiment analysis.
- **Issue Trends**: Analyzes issues reported in a repository, including open and closed issues, labels, and comments.
- **Tech Stack Identification**: Identifies the technologies demanded in issues and comments.
- **Issue Recommendations**: Provides recommendations for resolving issues using NLP techniques.
- **Predictive Analytics**: Predicts issue priorities using MindsDB.

## Requirements

- Python 3.7 or higher
- Required Python libraries:
  - `streamlit`
  - `requests`
  - `pandas`
  - `transformers`
  - `mindsdb_sdk`
  - `sqlite3`
  - `torch`

## Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/Adity20/github-analytics.git
    cd github-analytics
    ```

2. **Create a virtual environment and activate it**:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required libraries**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up MindsDB**:

    - Sign up for a MindsDB account and obtain your API key.
    - Replace `'your_mindsdb_api_key'` in the script with your actual MindsDB API key.

## Usage

1. **Run the Streamlit app**:

    ```sh
    streamlit run main.py
    ```

2. **Enter the GitHub username and repository name** in the sidebar to view analytics and insights.

## Project Structure

```plaintext
github-analytics-dashboard/
├── main.py            # Main script for the Streamlit app
├── requirements.txt   # Required Python libraries
├── readme.md          # Project documentation
└── db.db     # SQLite database for storing issues (generated during runtime)
