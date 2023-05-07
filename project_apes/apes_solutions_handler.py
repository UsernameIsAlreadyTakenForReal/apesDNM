import sys
from apes_static_definitions import Shared_Definitions

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


# with open("type_ekg/apes_solution_ekg_1_do.py") as f:
#     exec(f.read())

from type_ekg.iar import solution_ekg_1
from helpers_aiders_and_conveniencers.logger import Logger

shared_definitions = Shared_Definitions()
Logger = Logger()
test_solution = solution_ekg_1(shared_definitions, Logger)
