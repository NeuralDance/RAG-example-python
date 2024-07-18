import subprocess

def get_current_commit():
    try:
        # Run the Git command to get the current commit hash
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check for errors
        if result.returncode != 0:
            print("Error retrieving commit hash:", result.stderr)
            return None
        
        # Return the commit hash
        return result.stdout.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

current_git = get_current_commit()
print(current_git)