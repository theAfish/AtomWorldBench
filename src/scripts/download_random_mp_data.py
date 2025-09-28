import argparse
import os
import random
import sys
from mp_api.client import MPRester

def download_random_mp_cif(api_key, out_path, min_natoms=10, max_natoms=100, num_entries=10):
    """
    Download random materials CIFs from the Materials Project with conditions.

    Parameters:
        api_key (str): Your Materials Project API key.
        out_path (str): Output directory for CIF files.
        min_natoms (int): Minimum number of atoms in structure.
        max_natoms (int): Maximum number of atoms in structure.
        num_entries (int): Number of random entries to download.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with MPRester(api_key) as mpr:
        print(f"Querying materials with {min_natoms}-{max_natoms} atoms...")
        results = mpr.summary.search(num_sites=(min_natoms, max_natoms), fields=["material_id"])
        material_ids = [r.material_id for r in results]
        print(f"Found {len(material_ids)} materials in range.")
        if len(material_ids) == 0:
            print("No materials found with the specified atom count range.")
            return []
        if len(material_ids) < num_entries:
            print(f"Only found {len(material_ids)} materials. Reducing num_entries to {len(material_ids)}.")
            num_entries = len(material_ids)
        selected_ids = random.sample(material_ids, num_entries)
        downloaded = []
        for i, mid in enumerate(selected_ids, 1):
            try:
                structure = mpr.get_structure_by_material_id(mid)
                cif_filename = os.path.join(out_path, f"{mid}.cif")
                structure.to(filename=cif_filename, fmt="cif")
                print(f"[{i}/{num_entries}] Saved {cif_filename}")
                downloaded.append(cif_filename)
            except Exception as e:
                print(f"Failed to download {mid}: {e}")
        print(f"Downloaded {len(downloaded)} CIF files to {out_path}.")
        return downloaded

def main():
    parser = argparse.ArgumentParser(description="Download random CIFs from Materials Project.")
    parser.add_argument("--api_key", type=str, default=None, help="Materials Project API key. Can also set MP_API_KEY env variable.")
    parser.add_argument("--out_path", type=str, default="init_cifs", help="Output directory for CIF files.")
    parser.add_argument("--min_natoms", type=int, default=10, help="Minimum number of atoms in structure.")
    parser.add_argument("--max_natoms", type=int, default=100, help="Maximum number of atoms in structure.")
    parser.add_argument("--num_entries", type=int, default=1000, help="Number of random entries to download.")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        print("Error: API key must be provided via --api_key or MP_API_KEY environment variable.")
        sys.exit(1)

    download_random_mp_cif(
        api_key=api_key,
        out_path=args.out_path,
        min_natoms=args.min_natoms,
        max_natoms=args.max_natoms,
        num_entries=args.num_entries,
    )

if __name__ == "__main__":
    main()