import multiprocessing
import subprocess

# def run_script(script_name):
#     """Function to run a script using subprocess"""
#     subprocess.run(["python", script_name], check=True)

# # List of scripts to run in parallel
# scripts = ["script1.py", "script2.py", "script3.py"]

# # Create a process for each script
# processes = [multiprocessing.Process(target=run_script, args=(script,)) for script in scripts]

# # Start all processes
# for p in processes:
#     p.start()