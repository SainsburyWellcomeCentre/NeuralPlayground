from sehec.experimentconfig import custom_classes


def import_classes():
    import_list = []
    import_list.append("from sehec.agents.weber_2018 import ExcInhPlasticity")
    import_list.append("from sehec.arenas.simple2d import BasicSargolini2006")
    import_list.append("from sehec.arenas.simple2d import Sargolini2006")
    import_list += custom_classes.custom_classes_path
    return import_list