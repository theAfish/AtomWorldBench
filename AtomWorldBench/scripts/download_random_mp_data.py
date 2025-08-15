from mp_api.client import MPRester
import random
import os

def download_random_mp_cif(api_key, out_path, min_natoms=10, max_natoms=100, num_entries=10):
    """
    Download random materials cif from the Materials Project with conditions.

    Parameters:
    - api_key (str): Your Materials Project API key.
    - num_entries (int): Number of random entries to download.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with MPRester(api_key) as mpr:
        # Get all materials IDs with at least min_natoms atoms
        results = mpr.summary.search(num_sites=(min_natoms, max_natoms), fields=["material_id"])
        material_ids = [r.material_id for r in results]
        if len(material_ids) < num_entries:
            print(f"Only found {len(material_ids)} materials with at least {min_natoms} atoms.")
            num_entries = len(material_ids)
        selected_ids = random.sample(material_ids, num_entries)
        for mid in selected_ids:
            structure = mpr.get_structure_by_material_id(mid)
            cif_filename = os.path.join(out_path, f"{mid}.cif")
            structure.to(filename=cif_filename, fmt="cif")



if __name__ == "__main__":
    # Example
    api_key = "YOUR_API_KEY"
    download_random_mp_cif(api_key, 'init_cifs', min_natoms=10, max_natoms=100, num_entries=1000)