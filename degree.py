from typing import List, Tuple


def count(l: List[bool]):
    return sum(1 for x in l if x)

def kth_erdos_gallai(degrees: List[int], k: int):
    left = sum(degrees[:k]) 
    right = k*(k-1) + sum(min(x, k) for x in degrees[k:])
    return left, right

def is_graphic(l: List[int]):
    if not isinstance(l, list): return False
    if not all(isinstance(x, int) for x in l): return False
    
    # erdos gallai theorem (1960)
    l = sorted(l, reverse=True)
    if not l[-1] >= 0: return False
    if not len(l)-1 >= l[0]-1: return False
    if not sum(l) % 2 == 0: return False
    for k in range(1, len(l)):
        left, right = kth_erdos_gallai(l, k)
        if left > right:
            return False
    
    return True

def prepare_degrees(l: List[int]):
    assert is_graphic(l)
    return sorted(l, reverse=True)

def conjugate(l: List[int]):
    degrees = prepare_degrees(l)
    return [count(di >= j-1 for di in degrees[:j-1]) + count(di >= j for di in degrees[j:]) for j in range(1, len(l)+1)]

def avg_degree(l: List[int]):
    l = prepare_degrees(l)
    # 2m/n
    return sum(l)/len(l)

def density(l: List[int]):
    if len(l) - 1 <= 0: return float('inf')
    return avg_degree(l)/(len(l)-1)

# durfee number
def h_index(l: List[int]):
    degrees = prepare_degrees(l)
    return max(i for i in range(1,len(degrees)+1) if degrees[i-1] >= i-1)

def splittance(l: List[int]):
    degrees = prepare_degrees(l)
    h = h_index(degrees)
    return 1/2 * (h*(h-1) + sum(degrees[h:]) - sum(degrees[:h]))

def erdos_gallai_all(l: List[int]):
    degrees = prepare_degrees(l)
    for k in range(1, len(l)):
        left, right = kth_erdos_gallai(degrees, k)
        print(f'k={k}, {left:3}  <=  {right:3}  {"[eq]" if left == right else ""}')

def threshold_gap(l: List[int]):
    degrees = prepare_degrees(l)
    h = h_index(degrees)
    co = conjugate(degrees)
    return 1/2 * sum(abs(co[i] - degrees[i]) for i in range(h))

def is_split(l: List[int]):
    return splittance(l) == 0

def is_threshold(l: List[int]):
    return threshold_gap(l) == 0

def count_edges(l: List[int]):
    degrees = prepare_degrees(l)
    return sum(degrees)//2


# ⟨deg⟩ = avergae degree
# ⟨cc⟩ = cluster coefficient
# ⟨dist⟩ = characteristic path length

def main(data: List[int]):
    if not is_graphic(data):
        print("Data is not graphic")
        exit(1)
    fs = [
        prepare_degrees,
        count_edges,
        conjugate,
        avg_degree,
        density,
        h_index,
        splittance,
        threshold_gap,
        is_split,
        is_threshold,
        erdos_gallai_all
    ]
    for f in fs:
        print(f.__name__)
        print(f'-->  {f(data)}')
        print()


if __name__ == "__main__":
    # data = [3, 5, 2, 4, 3, 7, 2, 2]
    data = [7, 4, 2, 6, 5, 9, 0, 5, 8, 7, 1]
    main(data)