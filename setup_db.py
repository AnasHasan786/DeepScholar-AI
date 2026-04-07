import os
import sys

sys.path.append(os.path.join(os.getcwd(), "ra_core"))

from langgraph.checkpoint.postgres import PostgresSaver
from config.settings import DB_URI

def run_setup():
    print(f"Connecting to: {DB_URI}")
    try:
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()
        print("✅ Success: 'checkpoints' and 'writes' tables created.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_setup()