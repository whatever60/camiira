"""CLI entry points for CAMII."""

def detect_colonies_main():
    """Entry point for detect_colonies CLI."""
    from camiira.detect_colonies import main
    main()

def calc_calib_params_main():
    """Entry point for calc_calib_params CLI."""
    from camiira.calc_calib_params import main
    main()

def correct_coords_main():
    """Entry point for correct_coords CLI."""
    from camiira.correct_coords import main
    main()

def data_transform_main():
    """Entry point for data_transform CLI."""
    from camiira.data_transform import main
    main()

def draw_interaction_main():
    """Entry point for draw_interaction CLI."""
    from camiira.draw_interaction import main
    main()

def isolate_interaction_main():
    """Entry point for isolate_interaction CLI."""
    from camiira.isolate_interaction import main
    main()

def select_colonies_main():
    """Entry point for select_colonies CLI."""
    from camiira.select_colonies import main
    main()
