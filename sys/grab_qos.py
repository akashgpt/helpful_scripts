import subprocess
import re

def parse_time_limit(time_limit_str):
    """
    Convert a time limit string in the format [D-]HH:MM:SS to total hours.
    
    Args:
        time_limit_str (str): Time limit string (e.g., '01:30:00', '2-12:00:00').
    
    Returns:
        float: Total hours.
    """
    match = re.match(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", time_limit_str)
    if not match:
        return 0
    days, hours, minutes, seconds = match.groups(default=0)
    total_hours = int(days) * 24 + int(hours) + int(minutes) / 60 + int(seconds) / 3600
    return total_hours

def get_jobs_by_qos(user):
    """
    Categorize jobs based on their time limits using the `squeue` command.
    
    Args:
        user (str): Username to filter jobs.
    
    Returns:
        dict: A dictionary with QoS categories as keys and job counts as values.
    """
    try:
        # Run the squeue command with TIME_LIMIT included
        result = subprocess.run(
            ["squeue", "-u", user, "-o", "%.18i %.8u %.10l"],
            capture_output=True, text=True, check=True
        )
        # Skip the header and parse job time limits
        lines = result.stdout.splitlines()[1:]
        qos_counts = {"test": 0,"vshort": 0, "short": 0, "medium": 0, "long": 0}

        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                time_limit_str = parts[-1]  # The TIME_LIMIT column
                total_hours = parse_time_limit(time_limit_str)

                # print(f"Job ID: {parts[0]}, User: {parts[1]}, Time Limit: {time_limit_str}, Total Hours: {total_hours}")
                
                # Categorize based on the time limit
                if total_hours <= 1:
                    qos_counts["test"] += 1
                elif 1 < total_hours <= 5:
                    qos_counts["vshort"] += 1
                elif 5 < total_hours <= 24:
                    qos_counts["short"] += 1
                elif 24 < total_hours <= 72:
                    qos_counts["medium"] += 1
                elif 72 < total_hours < 144:
                    qos_counts["long"] += 1

        return qos_counts

    except FileNotFoundError:
        print("Error: 'squeue' command not found. Make sure Slurm is installed.")
        return {"test": 0,"vshort": 0, "short": 0, "medium": 0, "long": 0}
    except subprocess.CalledProcessError as e:
        print(f"Error while running 'squeue': {e.stderr}")
        return {"test": 0,"vshort": 0, "short": 0, "medium": 0, "long": 0}

# Usage
import os
user = os.getenv("USER", "your_username")  # Replace "your_username" if not using $USER
job_counts = get_jobs_by_qos(user)

for qos, count in job_counts.items():
    print(f"QoS: {qos}, Jobs: {count}")
