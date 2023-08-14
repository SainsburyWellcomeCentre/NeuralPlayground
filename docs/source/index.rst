.. NeuralPlayground documentation master file, created by
   sphinx-quickstart on Fri Dec  9 14:12:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NeuralPlayground's documentation!
=========================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api_index

By default the documentation includes the following sections:

* Getting started. Here you could describe the basic functionalities of your package. To modify this page, edit the file ``docs/source/getting_started.md``.
* API: here you can find the auto-generated documentation of your package, which is based on the docstrings in your code. To modify which modules/classes/functions are included in the API documentation, edit the file ``docs/source/api_index.rst``.

You can create additional sections with narrative documentation,
by adding new ``.md`` or ``.rst`` files to the ``docs/source`` folder.
These files shoult start with a level-1 (H1) header,
which will be used as the section title. Sub-sections can be created
with lower-level headers (H2, H3, etc.) within the same file.

To include a section in the rendered documentation,
add it to the ``toctree`` directive in this (``docs/source/index.rst``) file.

For example, you could create a ``docs/source/installation.md`` file
and add it to the ``toctree`` like this:

.. code-block:: rst

   .. toctree::
      :maxdepth: 2
      :caption: Contents:

      getting_started
      installation
      api_index

Index & Search
--------------
* :ref:`genindex`
* :ref:`search`
