from sehec.experimentconfig import custom_classes


def import_classes():
    from sehec.models.weber_and_sprekeler import ExcInhPlasticity
    from sehec.envs.arenas.simple2d import BasicSargolini2006
    for imp in custom_classes.custom_classes_path:
        eval(imp)