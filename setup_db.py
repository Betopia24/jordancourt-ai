import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

def setup_database():
    connection_string = os.getenv("POSTGRES_URL")
    if not connection_string:
        raise ValueError("POSTGRES_URL not found in .env")
    
    # Initialize the checkpointer to create tables
    checkpointer = PostgresSaver.from_conn_string(connection_string)
    checkpointer.setup()  # This creates the necessary tables
    print("Database setup complete.")

if __name__ == "__main__":
    setup_database()
