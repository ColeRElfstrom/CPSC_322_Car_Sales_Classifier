import copy
import csv
from mysklearn import myutils

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        try:
            index = self.column_names.index(col_identifier)
        except:
            ValueError
        col = []
        if include_missing_values:
            for value in self.data:
                col.append(value[index])
        else:
            for value in self.data:
                if value != "NA":
                    col.append(value[index])
        return col
    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        converted = 0
        for row in self.data:
            for value in row:
                try:
                    new_val = float(value)
                    converted = 1
                except:
                    pass
                if converted:
                    self.data[self.data.index(row)][row.index(value)] = new_val
                    converted = 0

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        i = 0
        for value in row_indexes_to_drop:
            try:
                self.data.pop(value - i)
                i += 1
            except:
                IndexError

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        in_file = open(filename, 'r')
        reader = csv.reader(in_file)
        i = 0
        for row in reader:
            if i == 0:
                self.column_names = row
                i = 1
            else:
                self.data.append(row)
        self.convert_to_numeric()
        in_file.close()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        out_file = open(filename, 'w')
        writer = csv.writer(out_file)
        writer.writerow(self.column_names)
        for row in self.data:
            writer.writerow(row)
        out_file.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        dupes = []
        tested = []
        index = 0
        row_vals = []
        for row in self.data:
            for value in key_column_names:
                row_vals.append(row[self.column_names.index(value)])
            if row_vals not in tested:
                tested.append(row_vals)
            elif row_vals in tested:
                dupes.append(index)
            index += 1
            row_vals = []

        return dupes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        idxs = []
        row_idx = 0
        i = 0
        for row in self.data:
            for value in row:
                if value == "":
                    idxs.append(row_idx)
            row_idx += 1
        for value in idxs:
            self.data.pop(value - i)
            i += 1

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        value_list = []
        index = self.column_names.index(col_name)
        for row in self.data:
            if isinstance(row[index], (float, int)):
                value_list.append(row[index])
        avg = sum(value_list) / len(value_list)
        for row in self.data:
            for value in row:
                if value == "NA":
                    self.data[self.data.index(row)][row.index(value)] = avg


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        header =  ["attribute", "min", "max", "mid", "avg", "median"]
        table = []
        if len(self.data) == 0:
            return MyPyTable(header, table)
        for col_name in col_names:
            calc_list = []
            value_list = []
            value_list.append(col_name)
            index = self.column_names.index(col_name)
            # extracting the values for calculation
            for row in self.data:
                if row[index] != "NA":
                    calc_list.append(row[index])
            # appending all of the stats in order
            value_list.append(min(calc_list))
            value_list.append(max(calc_list))
            calc_list.sort()
            mid = len(calc_list) // 2
            value_list.append((max(calc_list) + min(calc_list)) / 2)
            value_list.append(sum(calc_list) / len(calc_list))
            value_list.append((calc_list[mid] + calc_list[~mid]) / 2)
            table.append(value_list)

        return MyPyTable(header, table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # joins the headers together
        def join_header(header, header2):
            joined_header = header.copy()
            for val in header2:
                if val not in joined_header:
                    joined_header.append(val)
            return joined_header
        #joins the rows together
        def join_row(row, row2, header):
            joined_row = ["NA"] * len(header)
            for val in header:
                idx = header.index(val)
                if val in self.column_names:
                    idx_row = self.column_names.index(val)
                    joined_row[idx] = (row[idx_row])
                if val in other_table.column_names:
                    idx_row = other_table.column_names.index(val)
                    joined_row[idx] = (row2[idx_row])
            return joined_row

        table3 = []
        joined_row = []
        row_vals = []
        row2_vals = []
        header = join_header(self.column_names, other_table.column_names)
        for row in self.data:
            for value in key_column_names:
                row_vals.append(row[self.column_names.index(value)])
            for row2 in other_table.data:
                for value2 in key_column_names:
                    row2_vals.append(row2[other_table.column_names.index(value2)])
                if row_vals == row2_vals:
                    joined_row = join_row(row, row2, header)
                    table3.append(joined_row)
                row2_vals = []
            row_vals = []
        return MyPyTable(join_header(self.column_names, other_table.column_names), table3)

    def perform_full_outer_join(self, other_table, key_column_names): # currently just an inner join
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # joins the headers together
        def join_header(header, header2):
            joined_header = header.copy()
            for val in header2:
                if val not in joined_header:
                    joined_header.append(val)
            return joined_header
        #joins the rows together
        def join_row(row, row2, header):
            joined_row = ["NA"] * len(header)
            for val in header:
                idx = header.index(val)
                if val in self.column_names:
                    idx_row = self.column_names.index(val)
                    joined_row[idx] = (row[idx_row])
                if val in other_table.column_names:
                    idx_row = other_table.column_names.index(val)
                    joined_row[idx] = (row2[idx_row])
            return joined_row

        def join_row_missing(row, side, header):
            joined_row = ["NA"] * len(header)
            for val in header:
                idx = header.index(val)
                if side == "left":
                    if val in self.column_names:
                        idx_row = self.column_names.index(val)
                        joined_row[idx] = (row[idx_row])
                else:
                    if val in other_table.column_names:
                        idx_row = other_table.column_names.index(val)
                        joined_row[idx] = (row2[idx_row])
            return joined_row

        table3 = []
        joined_row = []
        matched = 1
        header = join_header(self.column_names, other_table.column_names)
        for row in self.data:
            row_vals = []
            for value in key_column_names:
                row_vals.append(row[self.column_names.index(value)])
            for row2 in other_table.data:
                row2_vals = []
                for value2 in key_column_names:
                    row2_vals.append(row2[other_table.column_names.index(value2)])
                if row_vals == row2_vals:
                    joined_row = join_row(row, row2, header)
                    table3.append(joined_row)
                    matched = 0
            if matched:
                joined_row = join_row_missing(row, "left", header)
                table3.append(joined_row)
            matched = 1
        table_val = []
        for row in table3:
            other_val = []
            for val in key_column_names:
                other_val.append(row[header.index(val)])
            table_val.append(other_val)
        for row2 in other_table.data:
            row_val = []
            for val in key_column_names:
                row_val.append(row2[other_table.column_names.index(val)])
            if row_val not in table_val:
                table3.append(join_row_missing(row2, "right", header))
        return MyPyTable(join_header(self.column_names, other_table.column_names), table3)
