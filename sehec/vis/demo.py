from random import randint
from datetime import date
from bokeh.models import ColumnDataSource, TableColumn, DateFormatter, DataTable
from bokeh.layouts import column
from bokeh.models.widgets import TextInput
from bokeh.plotting import curdoc

data = dict(dates = [date(2014, 3, i + 1) for i in range(10)],
            downloads = [randint(0, 100) for i in range(10)],
            identities = ['id_' + str(x) for x in range(10)])

source = ColumnDataSource(data)

columns = [TableColumn(field = "dates", title = "Date",
           formatter = DateFormatter()),
           TableColumn(field = "downloads", title = "Downloads")]

data_table = DataTable(source = source, columns = columns, width = 280, height = 280, editable = True)
table_row = TextInput(value = '', title = "Row index:")
table_cell_column_1 = TextInput(value = '', title = "Date:")
table_cell_column_2 = TextInput(value = '', title = "Downloads:")

def function_source(attr, old, new):
    try:
        selected_index = source.selected.indices[0]
        table_row.value = str(selected_index)
        table_cell_column_1.value = str(source.data["dates"][selected_index])
        table_cell_column_2.value = str(source.data["downloads"][selected_index])
    except IndexError:
        pass

source.selected.on_change('indices', function_source)
curdoc().add_root(column(data_table, table_row, table_cell_column_1, table_cell_column_2))