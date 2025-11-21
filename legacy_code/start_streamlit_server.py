import subprocess
import os

stream = os.popen('streamlit run main.py')

# process = subprocess.Popen(['streamlit', 'run',  './main.py'],
#                            stdout=subprocess.PIPE,
#                            universal_newlines=True)