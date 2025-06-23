import h5py
import os

names = [
    'change_atom_action',
    'delete_around_atom_action', 
    'delete_below_atom_action', 
    'insert_between_atoms_action',
    'move_around_atom_action',
    'move_atom_action',
    'move_selected_atoms_action',
    'move_towards_atom_action',
    'remove_atom_action',
    'rotate_around_atom_action',
    'swap_atoms_action'
]

for name in names:
    cif_folder = f'D:/Codes/AtomWorld/src/data/output_cifs/{name}'
    cif_files_list = [os.path.join(cif_folder, f) for f in os.listdir(cif_folder) if f.lower().endswith('.cif')]

    hdf5_output_path = f'{name}.hdf5'

    with h5py.File(hdf5_output_path, 'w') as f:
        for cif_path in cif_files_list:
            filename = os.path.basename(cif_path)
            with open(cif_path, 'r') as cif_f:
                cif_content = cif_f.read()
                f.create_dataset(filename, data=cif_content)

    print(f"HDF5 file '{hdf5_output_path}' created successfully.")