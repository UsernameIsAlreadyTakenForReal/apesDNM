# This file:
# Is meant to call the suitable solution (by user input or by apes's recommendation).

# import sys
# from apes_static_definitions import Shared_Definitions

# ##### Arguments
# n = len(sys.argv)
# print("Total number of arguments passed: ", n)

# print("\nName of Python script:", sys.argv[0])

# print("\nArguments passed:", end=" ")
# for i in range(1, n):
#     print(sys.argv[i], end=" ")

# Sum = 0
# # Using argparse module
# for i in range(1, n):
#     Sum += int(sys.argv[i])

# print("\n\nResult:", Sum)

##### open as if concatenated to main
# with open("type_ekg/apes_solution_ekg_1_do.py") as f:
#     exec(f.read())

import apes_dataset_handler


example_path = "../../Dataset - ECG_Heartbeat/mitbih_test.csv"
pandas_df = apes_dataset_handler.process_file_type(example_path)

print(pandas_df)
