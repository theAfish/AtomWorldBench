class Atom:
    def __init__(self, name, atomic_number):
        self.name = name
        self.atomic_number = atomic_number

    def __repr__(self):
        return f"Atom(name={self.name}, atomic_number={self.atomic_number})"

def get_atom_info(atomic_number):
    periodic_table = {
        1: Atom("Hydrogen", 1),
        2: Atom("Helium", 2),
        3: Atom("Lithium", 3),
        4: Atom("Beryllium", 4),
        5: Atom("Boron", 5),
        6: Atom("Carbon", 6),
        7: Atom("Nitrogen", 7),
        8: Atom("Oxygen", 8),
        9: Atom("Fluorine", 9),
        10: Atom("Neon", 10),
    }
    return periodic_table.get(atomic_number, None)