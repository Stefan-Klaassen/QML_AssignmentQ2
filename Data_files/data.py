from pathlib import Path

def read_file(file: Path, header: list[str]) -> list[dict[str, int]]:
    data = []
    with file.open('r') as f:
        for line in f:
            row = line.strip().split("\t")
            row_dict = dict(zip(header, row))
            data.append(row_dict)
    return data

header_data = ['LOC_ID', 'XCOORD', 'YCOORD', 'DEMANDh location', 'READYTIMEf pickup', 'DUETIMEpickup', 'SERVICETIME', 'CHARGING']
header_periodsCharge = ['PER_ID', 'STARTTIME', 'ENDTIME', 'COST']

file_data_small = Path('.') / 'Data_files' / 'data_small.txt'
data_small = read_file(file=file_data_small, header=header_data)

file_data_large = Path('.') / 'Data_files' / 'data_large.txt'
data_large = read_file(file=file_data_large, header=header_data)

file_data_periodsCharge = Path('.') / 'Data_files' / 'data_PeriodsCharge.txt'
data_periodsCharge = read_file(file=file_data_periodsCharge, header=header_periodsCharge)

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

