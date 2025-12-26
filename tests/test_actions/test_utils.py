from AtomWorldBench.atom_world.actions.motif_actions.utils import get_random_motif


def test_get_random_motif(orig_atoms):
    for motif_alias in ["cluster", "site", "sphere", "box", "bond"]:
        motif = get_random_motif(motif_alias, orig_atoms, seed=123)
        assert motif is not None
        assert motif.__class__.__name__.lower().startswith(motif_alias)
