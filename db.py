import sqlite3
import os

def get_db_connection():
    db_path = os.path.abspath(os.getcwd()) + "/github_issues.db"
    connection = sqlite3.connect(db_path)
    return connection

def create_tables():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS issues_table (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issue_number INTEGER,
        title TEXT,
        description TEXT,
        recommendation TEXT
    )
    """)
    connection.commit()
    connection.close()

def insert_issue_data(issue_number, title, description, recommendation):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
    INSERT INTO issues_table (issue_number, title, description, recommendation)
    VALUES (?, ?, ?, ?)
    """, (issue_number, title, description, recommendation))
    connection.commit()
    connection.close()

def get_all_issues_data():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM issues_table")
    rows = cursor.fetchall()
    connection.close()
    return rows

def get_issue_by_number(issue_number):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM issues_table WHERE issue_number = ?", (issue_number,))
    row = cursor.fetchone()
    connection.close()
    return row

def update_recommendation(issue_number, recommendation):
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("""
    UPDATE issues_table
    SET recommendation = ?
    WHERE issue_number = ?
    """, (recommendation, issue_number))
    connection.commit()
    connection.close()
