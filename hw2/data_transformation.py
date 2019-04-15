class DataTransformation:
    def __init__(self, description, input_column_names, output_column_name, function):
        self.description = description
        self.input_column_names = input_column_names
        self.output_column_name = output_column_name
        self.function = function

    def __call__(self, dataframe):
        return self.function(dataframe[self.input_column_names])

def discretize():
    pass

def binarize(): # is this a word? merriam webster says so
    pass
