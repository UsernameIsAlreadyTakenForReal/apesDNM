class plm:
    def __init__(self):
        self.name = "hai sa schimbam sa nu fie acelasi ca calasa"
        self.cacat = "da"
        pass
    
    

import pandas as pd
example_path = "../Dataset - ECG_Heartbeat/mitbih_test - HAAAAAAAAAA.csv"
# example_path = "../Dataset - ECG_Heartbeat/mitbih_test.csv"
# df = pd.read_csv(example_path, header=None)
# df = pd.read_csv(example_path)

try:
    df = pd.read_csv(example_path, header=None)
except:
    print("fuck me")
else:
    print("pe bune?")
finally:
    print("retard")