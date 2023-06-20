class NPGConfig(object):
    """Base class for config objects

    Methods
    -------
    __str__(self, indent=0)
        Print the config object in a readable way
    keys(self)
        Return keys of the config object
    values(self)
        Return values of the config object
    items(self)
        Return items of the config object
    """

    def __str__(self, indent=0):
        """Print the config object in a readable way
        Parameters
        ----------
        indent: int
            Indentation level for nested config objects

        Returns
        -------
        show_str: str
            String containing the config object in a readable way
        """
        show_str = ""
        for key, value in self.__dict__.items():
            if isinstance(value, NPGConfig):
                show_str += f"{key}:\n"
                show_str += value.__str__(indent=indent + 1)
            else:
                show_str += " " * indent * 4 + f"{key}: {value}\n"
        return show_str

    def keys(self):
        """Return keys of the config object"""
        return self.__dict__.keys()

    def values(self):
        """Return values of the config object"""
        return self.__dict__.values()

    def items(self):
        """Return items of the config object"""
        return self.__dict__.items()
