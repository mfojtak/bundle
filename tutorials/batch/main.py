import os
jobs = os.environ["JOBS"]
job = os.environ["JOB"]

print("Job {}/{} finished".format(job, jobs))