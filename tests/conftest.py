"""
Pytest configuration for database tests.
"""

import os
import pytest
from parkiet.database.connection import DatabaseConnection


@pytest.fixture(scope="session")
def test_db_connection():
    """
    Create a test database connection using environment variables.
    Set these environment variables for testing:
    - DB_HOST=localhost
    - DB_PORT=5433  # Different port for test database
    - DB_NAME=parkiet_test
    - DB_USER=parkiet
    - DB_PASSWORD=your_password
    """
    # Override environment variables for testing
    test_env = {
        "DB_HOST": os.getenv("TEST_DB_HOST", "localhost"),
        "DB_PORT": os.getenv("TEST_DB_PORT", "5433"),
        "DB_NAME": os.getenv("TEST_DB_NAME", "parkiet_test"),
        "DB_USER": os.getenv("TEST_DB_USER", "parkiet"),
        "DB_PASSWORD": os.getenv(
            "TEST_DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "parkiet")
        ),
    }

    # Set environment variables for this test session
    for key, value in test_env.items():
        os.environ[key] = value

    # Create test database connection
    test_db = DatabaseConnection()

    # Verify connection works
    try:
        with test_db.get_cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as e:
        pytest.fail(f"Test database connection failed: {e}")

    return test_db


@pytest.fixture
def use_test_database(test_db_connection):
    """
    Use test database for database tests.
    Include this fixture in test functions that need database access.
    """
    # Clear all tables before each test
    with test_db_connection.get_cursor() as cursor:
        cursor.execute("DELETE FROM chunk_event_links")
        cursor.execute("DELETE FROM audio_events")
        cursor.execute("DELETE FROM audio_chunks")
        cursor.execute("DELETE FROM audio_files")
        cursor.execute("DELETE FROM speakers")

    yield test_db_connection
