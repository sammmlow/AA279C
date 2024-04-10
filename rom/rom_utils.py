
import csv

def read_param_csv(file_name: str):
    """
    Abstract function to read a csv file and load in the keys as strings and
    the values as floats.
    """
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Ignore the first row of the csv file
        next(reader)

        output_dict = {}

        for row in reader:
            output_dict[row[0]] = float(row[1])

    return output_dict
