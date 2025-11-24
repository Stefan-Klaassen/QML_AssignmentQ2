import math
from data import Node

MAX_FLOAT = 1e6

def build_distance_mat(data: list[Node]) -> list[list[float]]:

    def _euclidean_distance(node1: Node, node2: Node) -> float:
        return math.sqrt( (node2.XCOORD - node2.XCOORD)**2 + (node2.YCOORD - node1.YCOORD)**2 )

    mat = []
    for i, start in enumerate(data):
        row = []
        for j, dist in enumerate(data):
            if i == j:
                row.append(MAX_FLOAT)
                continue
            row.append(_euclidean_distance(start, dist))
        mat.append(row)
    return mat

if __name__ == "__main__":
    from data import data_small as data
    mat = build_distance_mat(data)
    for row in mat:
        print(row)

