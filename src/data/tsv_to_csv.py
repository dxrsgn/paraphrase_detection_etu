import sys
import pandas as pd


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Not enough args")
    tsv_file = sys.argv[1]
    csv_file = sys.argv[2]
    csv_table = pd.read_table(tsv_file, sep='\t')
    csv_table.to_csv(csv_file, index=False)