import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import re

# NOTE: Heavily based on (slightly adapted from) https://github.com/calvin-zcx/moflow

# Updated decoders and valency dictionary
atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {
    0: None,
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def flatten_graph_data(adj, x):
    # Reshape `adj` and `x` appropriately for the new shapes
    return torch.cat(
        (adj.reshape([adj.shape[0], -1]), x.reshape([x.shape[0], -1])), dim=1
    )


def split_channel(x):
    n = x.shape[1] // 2
    return x[:, :n], x[:, n:]


def get_graph_data(x, num_nodes, num_relations, num_features):
    """
    Converts a vector of shape [b, num_nodes, m] to adjacency matrix
    of shape [b, num_nodes, num_nodes, num_relations]
    and a feature matrix of shape [b, num_nodes, num_features].
    """
    adj = x[:, : num_nodes * num_nodes * num_relations].reshape(
        [-1, num_nodes, num_nodes, num_relations]
    )
    feat_mat = x[:, num_nodes * num_nodes * num_relations :].reshape(
        [-1, num_nodes, num_features]
    )
    return adj, feat_mat


def Tensor2Mol(A, x):
    mol = Chem.RWMol()
    # Convert adjacency matrix and features into a molecule
    atoms = np.argmax(x, axis=2)  # Shape: (batch_size, 9)
    batch_size = atoms.shape[0]
    molecules = []

    for b in range(batch_size):
        mol = Chem.RWMol()
        atoms_exist = atoms[b] != 4  # 4 represents 'no atom'
        atom_indices = np.nonzero(atoms_exist)[0]
        for idx in atom_indices:
            mol.AddAtom(Chem.Atom(atom_decoder_m[atoms[b, idx]]))

        adj_matrix = np.argmax(A[b], axis=-1)  # Shape: (9, 9)
        adj_matrix = adj_matrix[atoms_exist][:, atoms_exist]
        for start, end in zip(*np.nonzero(adj_matrix)):
            if start > end:
                bond_type = bond_decoder_m.get(adj_matrix[start, end], None)
                if bond_type:
                    mol.AddBond(int(start), int(end), bond_type)

        molecules.append(mol)

    return molecules


def construct_mol(x, A, atomic_num_list):
    """
    :param x: (batch_size, 9, 4) Atom one-hot encodings
    :param A: (batch_size, 9, 9, 5) Adjacency tensor with bond types
    :param atomic_num_list: List of atomic numbers
    :return: A list of RDKit Mol objects
    """
    batch_size = x.shape[0]
    molecules = []

    for b in range(batch_size):
        mol = Chem.RWMol()
        atoms = np.argmax(x[b], axis=1)
        atoms_exist = atoms != len(atomic_num_list) - 1
        atom_indices = np.nonzero(atoms_exist)[0]

        for idx in atom_indices:
            mol.AddAtom(Chem.Atom(int(atomic_num_list[atoms[idx]])))

        adj_matrix = np.argmax(A[b], axis=-1)  # Shape: (9, 9)
        adj_matrix = adj_matrix[atoms_exist][:, atoms_exist]
        for start, end in zip(*np.nonzero(adj_matrix)):
            if start > end:
                bond_type = bond_decoder_m.get(adj_matrix[start, end], None)
                if bond_type:
                    mol.AddBond(int(start), int(end), bond_type)

        molecules.append(mol)

    return molecules


def construct_mol_with_validation(x, A, atomic_num_list):
    mol = Chem.RWMol()
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            t = adj[start, end]
            while not valid_mol_can_with_seg(mol):
                mol.RemoveBond(int(start), int(end))
                t = t - 1
                if t >= 1:
                    mol.AddBond(int(start), int(end), bond_decoder_m[t])

    return mol


def valid_mol(x):
    s = (
        Chem.MolFromSmiles(Chem.MolToSmiles(x, isomericSmiles=True))
        if x is not None
        else None
    )
    if s is not None and "." not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and "." in sm:
        vsm = [(s, len(s)) for s in sm.split(".")]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find("#")
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r"\d+", e_sub)))
        return False, atomid_valence


def correct_mol(x):
    xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (
                        b.GetIdx(),
                        int(b.GetBondType()),
                        b.GetBeginAtomIdx(),
                        b.GetEndAtomIdx(),
                    )
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t in bond_decoder_m:
                    mol.RemoveBond(start, end)
                    if t >= 1:
                        mol.AddBond(start, end, bond_decoder_m[t])
                else:
                    print(
                        f"Invalid bond type 't' encountered: {t}. Skipping bond creation."
                    )
                    continue

    return mol


def test_correct_mol():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(7))
    mol.AddBond(0, 1, Chem.rdchem.BondType.DOUBLE)
    mol.AddBond(1, 2, Chem.rdchem.BondType.TRIPLE)
    mol.AddBond(0, 3, Chem.rdchem.BondType.TRIPLE)
    print(Chem.MolToSmiles(mol))  # C#C=C#N
    mol = correct_mol(mol)
    print(Chem.MolToSmiles(mol))  # C=C=C=N


def check_tensor(x):
    return valid_mol(Tensor2Mol(*x))


def adj_to_smiles(adj, x, atomic_num_list):
    valid = [
        Chem.MolToSmiles(
            construct_mol(x_elem, adj_elem, atomic_num_list), isomericSmiles=True
        )
        for x_elem, adj_elem in zip(x, adj)
    ]
    return valid


def check_validity(
    adj,
    x,
    atomic_num_list,
    return_unique=True,
    correct_validity=True,
    largest_connected_comp=True,
    debug=True,
):
    adj = _to_numpy_array(adj)
    x = _to_numpy_array(x)
    if correct_validity:
        valid = []

        mols = construct_mol(x, adj, atomic_num_list)
        for mol in mols:
            cmol = correct_mol(mol)
            vcmol = valid_mol_can_with_seg(
                cmol, largest_connected_comp=largest_connected_comp
            )
            valid.append(vcmol)
    else:
        valid = [
            valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
            for x_elem, adj_elem in zip(x, adj)
        ]  # len()=1000
    valid = [mol for mol in valid if mol is not None]  # len()=valid number, say 794
    if debug:
        print("valid molecules: {}/{}".format(len(valid), adj.shape[0]))
        for i, mol in enumerate(valid):
            print("[{}] {}".format(i, Chem.MolToSmiles(mol, isomericSmiles=False)))

    n_mols = x.shape[0]
    valid_ratio = len(valid) / n_mols  # say 794/1000
    valid_smiles = [Chem.MolToSmiles(mol, isomericSmiles=False) for mol in valid]
    unique_smiles = list(set(valid_smiles))  # unique valid, say 788
    unique_ratio = 0.0
    if len(valid) > 0:
        unique_ratio = len(unique_smiles) / len(valid)  # say 788/794
    if return_unique:
        valid_smiles = unique_smiles
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    abs_unique_ratio = len(unique_smiles) / n_mols
    if debug:
        print(
            "valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}%".format(
                valid_ratio * 100, unique_ratio * 100, abs_unique_ratio * 100
            )
        )
    results = dict()
    results["valid_mols"] = valid_mols
    results["valid_smiles"] = valid_smiles
    results["valid_ratio"] = valid_ratio * 100
    results["unique_ratio"] = unique_ratio * 100
    results["abs_unique_ratio"] = abs_unique_ratio * 100

    return results


def check_novelty(
    gen_smiles, train_smiles, n_generated_mols
):  # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.0
        abs_novel_ratio = 0.0
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        novel_ratio = novel * 100.0 / len(gen_smiles)  # 743*100/788=94.289
        abs_novel_ratio = novel * 100.0 / n_generated_mols
    print("novelty: {:.3f}%, abs novelty: {:.3f}%".format(novel_ratio, abs_novel_ratio))
    return novel_ratio, abs_novel_ratio


def _to_numpy_array(a):  # , gpu=-1):
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    # if gpu >= 0:
    #     return cuda.to_cpu(a)
    elif isinstance(a, np.ndarray):
        # We do not use cuda np.ndarray in pytorch
        pass
    else:
        raise TypeError("a ({}) is not a torch.Tensor".format(type(a)))
    return a


def save_mol_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


if __name__ == "__main__":

    test_correct_mol()
