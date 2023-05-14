import sys
from apes_static_definitions import Shared_Definitions

##### Arguments
n = len(sys.argv)
print("Total number of arguments passed: ", n)

print("\nName of Python script:", sys.argv[0])

print("\nArguments passed:", end=" ")
for i in range(1, n):
    print(sys.argv[i], end=" ")

Sum = 0
# Using argparse module
for i in range(1, n):
    Sum += int(sys.argv[i])

print("\n\nResult:", Sum)

##### open as if concatenated to main
# with open("type_ekg/apes_solution_ekg_1_do.py") as f:
#     exec(f.read())


##### open as classes
from type_ekg.apes_solution_ekg_1_train import solution_ekg_1
from helpers_aiders_and_conveniencers.logger import Logger

Logger = Logger()
shared_definitions = Shared_Definitions()

test_solution = solution_ekg_1(shared_definitions, Logger)

dataset = 0
