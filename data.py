from pathlib import Path
from dataclasses import dataclass

@dataclass
class Node:
    LOC_ID: int
    XCOORD: int
    YCOORD: int
    DEMAND: int
    READYTIME: int
    DUETIME: int
    SERVICETIME: int
    CHARGING: int

@dataclass
class PeriodsCharge:
    PER_ID: int
    STARTTIME: int
    ENDTIME: int
    COST: int

def read_file(file: Path, header: list[str], Cls):
    data = []
    with file.open('r') as f:
        for line in f:
            row = [int(v) for v in line.strip().split()]
            instance = Cls(*row)
            data.append(instance)
    return data

header_node = ['LOC_ID', 'XCOORD', 'YCOORD', 'DEMAND', 'READYTIME', 'DUETIME', 'SERVICETIME', 'CHARGING']
header_periodsCharge = ['PER_ID', 'STARTTIME', 'ENDTIME', 'COST']

file_data_small = Path('.') / 'Data_files' / 'data_small.txt'
data_small: list[Node] = read_file(file=file_data_small, header=header_node, Cls=Node)

file_data_large = Path('.') / 'Data_files' / 'data_large.txt'
data_large: list[Node] = read_file(file=file_data_large, header=header_node, Cls=Node)

file_data_periodsCharge = Path('.') / 'Data_files' / 'data_PeriodsCharge.txt'
data_periodsCharge: list[PeriodsCharge] = read_file(file=file_data_periodsCharge, header=header_periodsCharge, Cls=PeriodsCharge)



if __name__ == "__main__":
    from pprint import pprint
    
    print("data small:")
    pprint(data_small)
    print('\n'*2)

    print("data large:")
    pprint(data_large)
    print('\n'*2)

    print("data periods charge:")
    pprint(data_periodsCharge)
    print('\n'*2)
