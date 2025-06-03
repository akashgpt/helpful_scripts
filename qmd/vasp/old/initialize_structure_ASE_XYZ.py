from ase.io import read
import numpy as np

def get_minimum_interatomic_distance(poscar_filename):
    """
    Reads a POSCAR file and calculates the minimum interatomic distance,
    accounting for periodic boundary conditions.

    Args:
        poscar_filename (str): The path to the POSCAR file.

    Returns:
        float: The minimum interatomic distance.
               Returns float('inf') if fewer than 2 atoms are present.
    """
    try:
        # 1. Read the POSCAR file
        atoms = read(poscar_filename, format="vasp")
    except Exception as e:
        print(f"Error reading POSCAR file: {e}")
        return None

    num_atoms = len(atoms)

    # 2. Handle cases with fewer than 2 atoms
    if num_atoms < 2:
        print("Fewer than 2 atoms in the structure, cannot calculate interatomic distances.")
        return float('inf')

    # 3. Get all pairwise distances, accounting for PBC (mic=True means Minimum Image Convention)
    # This returns a symmetric matrix where all_distances[i, j] is the distance
    # between atom i and atom j. Diagonal elements (i=j) are 0.
    all_distances = atoms.get_all_distances(mic=True)

    # 4. Find the minimum distance
    # We need to find the minimum of the upper (or lower) triangle of this matrix
    # to avoid self-distances (diagonal) and duplicate pair considerations.
    min_dist = float('inf')
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms): # Only consider j > i
            if all_distances[i, j] < min_dist:
                min_dist = all_distances[i, j]
    
    # A more numpy-idiomatic way to get the minimum off-diagonal element:
    # Create a mask for the upper triangle, excluding the diagonal
    # indices = np.triu_indices_from(all_distances, k=1)
    # if all_distances[indices].size > 0:
    #    min_dist_numpy = np.min(all_distances[indices])
    # else:
    #    min_dist_numpy = float('inf') # Should be same as loop method

    return min_dist

if __name__ == '__main__':
    # Create a dummy POSCAR file for testing
#     poscar_content_valid = """\
# Test System
# 1.0
# 10.0  0.0  0.0
# 0.0  10.0  0.0
# 0.0  0.0  10.0
# Si
# 2
# Direct
# 0.0  0.0  0.0  Si_1
# 0.1  0.0  0.0  Si_2  # Distance = 1.0 Å
# """
    # with open("POSCAR_test", "w") as f:
    #     f.write(poscar_content_valid)

    min_d = get_minimum_interatomic_distance("POSCAR_random")
    if min_d is not None:
        print(f"Minimum interatomic distance (PBC accounted for): {min_d:.4f} Å")

#     poscar_content_single_atom = """\
# Test System Single
# 1.0
# 10.0  0.0  0.0
# 0.0  10.0  0.0
# 0.0  0.0  10.0
# Fe
# 1
# Direct
# 0.0  0.0  0.0
# """
#     with open("POSCAR_single", "w") as f:
#         f.write(poscar_content_single_atom)

#     min_d_single = get_minimum_interatomic_distance("POSCAR_single")
#     if min_d_single is not None:
#         print(f"Result for single atom structure: {min_d_single}") # Expected: inf

#     poscar_content_pbc_case = """\
# Test System PBC
# 1.0
# 10.0  0.0  0.0
# 0.0  10.0  0.0
# 0.0  0.0  10.0
# C
# 2
# Direct
# 0.05  0.0  0.0  C1
# 0.95  0.0  0.0  C2 # Direct distance = 9.0, PBC distance = 1.0
# """
#     with open("POSCAR_pbc", "w") as f:
#         f.write(poscar_content_pbc_case)
#     min_d_pbc = get_minimum_interatomic_distance("POSCAR_pbc")
#     if min_d_pbc is not None:
#         print(f"Minimum distance in PBC case: {min_d_pbc:.4f} Å") # Expected: 1.0