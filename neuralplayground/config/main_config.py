class NPGConfig(object):
    def __str__(self, indent=0):
        show_str = ""
        for key, value in self.__dict__.items():
            if isinstance(value, NPGConfig):
                show_str += f"{key}:\n"
                show_str += value.__str__(indent=indent + 1)
            else:
                show_str += " " * indent * 4 + f"{key}: {value}\n"
        return show_str

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()
