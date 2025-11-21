import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    try:
        import boat
        print("boat package imported successfully")
        from boat.core.engine import initiate_run
        print("boat.core.engine imported successfully")
        from boat.models import lstm_model
        print("boat.models imported successfully")
        from boat.api.app import app
        print("boat.api.app imported successfully")
        from boat.ui.streamlit_app import run_ui
        print("boat.ui.streamlit_app imported successfully")
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
