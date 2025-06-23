import h5py
import os


def load_cifs_from_hdf5(hdf5_filepath):
    """
    Loads all CIF content from an HDF5 file.
    Returns a dictionary where keys are original CIF filenames and values are CIF content strings.
    """
    cif_data = {}
    with h5py.File(hdf5_filepath, 'r') as f:
        for key in f.keys():
            if key.lower().endswith('.cif'): # Assuming you saved with .cif extension
                cif_content = f[key][()] # Read the dataset content
                # If you saved as bytes, decode it:
                if isinstance(cif_content, bytes):
                    cif_content = cif_content.decode('utf-8')
                cif_data[key] = cif_content
    return cif_data

if __name__ == "__main__":
    hdf5_file_path = 'add_atom_action.hdf5'
    cif_data = load_cifs_from_hdf5(hdf5_file_path)
    for filename, content in cif_data.items():
        print(f"Filename: {filename}")
        print(f"Content:\n{content[:100]}...")  # Print first 100 characters of each CIF content
        print("-" * 40)