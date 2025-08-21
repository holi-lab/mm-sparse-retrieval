# utils.py
import numpy as np
import scipy.stats as stats
import ir_measures
from ir_measures import *
from collections import defaultdict

def write_trec_file(trec_run, run_file_path):
    print(f"Saving TREC run file to {run_file_path}")
    with open(str(run_file_path), "w") as f:  # Convert Path to string
        for qid in trec_run:
            for rank, did in enumerate(trec_run[qid]):
                f.write(f"{qid} Q0 {did} {rank+1} {trec_run[qid][did]} lsr42\n")