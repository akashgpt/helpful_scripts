import numpy as np
from ase import Atoms
from ase.io import write


def random_rotation_matrix(rng):
    """Uniform random 3D rotation from a quaternion."""
    u1, u2, u3 = rng.random(3)

    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    R = np.array([
        [1 - 2*(q3*q3 + q4*q4),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [    2*(q2*q3 + q1*q4), 1 - 2*(q2*q2 + q4*q4),     2*(q3*q4 - q1*q2)],
        [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2*q2 + q3*q3)],
    ])
    return R


def ammonia_geometry(bond_length=1.02, angle_deg=106.7):
    """
    Idealized NH3 geometry centered on N at origin.
    Returns 3 H vectors in Angstrom.
    Molecular C3 axis is along +z before rotation.
    """
    theta = np.deg2rad(angle_deg)

    # For 3 equivalent H atoms at azimuths 0,120,240 and common polar angle alpha:
    # cos(theta_HNH) = 1.5 cos^2(alpha) - 0.5
    cos_alpha_sq = (np.cos(theta) + 0.5) / 1.5
    cos_alpha_sq = np.clip(cos_alpha_sq, 0.0, 1.0)
    cos_alpha = np.sqrt(cos_alpha_sq)
    alpha = np.arccos(cos_alpha)
    sin_alpha = np.sin(alpha)

    H = []
    for phi in [0.0, 2*np.pi/3, 4*np.pi/3]:
        vec = bond_length * np.array([
            sin_alpha * np.cos(phi),
            sin_alpha * np.sin(phi),
            cos_alpha,
        ])
        H.append(vec)

    return np.array(H)


def build_nh3_phase_iii_supercell(
    a=4.244,
    reps=(2, 2, 2),
    bond_length=1.02,
    angle_deg=106.7,
    seed=42,
):
    """
    Build a representative NH3-III supercell:
    - N on the FCC lattice
    - one NH3 molecule per N site
    - random molecular orientations
    """
    rng = np.random.default_rng(seed)

    # Conventional FCC basis for the cubic Fm-3m cell
    fcc_basis_frac = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])

    nx, ny, nz = reps
    cell = np.diag([a * nx, a * ny, a * nz])

    local__H_vectors = ammonia_geometry(
        bond_length=bond_length,
        angle_deg=angle_deg,
    )

    symbols = []
    positions = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_shift = np.array([i, j, k], dtype=float)

                for basis in fcc_basis_frac:
                    frac = (cell_shift + basis) / np.array([nx, ny, nz], dtype=float)
                    N_pos = frac @ cell

                    symbols.append("N")
                    positions.append(N_pos)

                    R = random_rotation_matrix(rng)
                    for hvec in local__H_vectors:
                        H_pos = N_pos + R @ hvec
                        symbols.append("H")
                        positions.append(H_pos)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.wrap()

    return atoms


if __name__ == "__main__":
    # 128 atoms = 32 NH3 molecules
    atoms = build_nh3_phase_iii_supercell(
        a=4.244,
        reps=(2, 2, 2),
        bond_length=1.02,
        angle_deg=106.7,
        seed=7,
    )

    print(atoms)
    print("Number of atoms:", len(atoms))

    # For VESTA / visualization
    write("NH3_III_128atoms.cif", atoms)

    # For VASP
    write("POSCAR_NH3_III_128atoms", atoms, direct=True, vasp5=True)