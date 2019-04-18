class Transformation:
    def __init__(self, name, input_column_names, output_column_name, function):
        self.name = name
        self.input_column_names = input_column_names
        self.output_column_name = output_column_name
        self.function = function

    def __call__(self, dataframe):
        return self.function(dataframe)

    def __repr__(self):
        return self.name

def replace_missing(column):
    input_col  = column
    output_col = column+"_clean"
    def replace(dataframe):
        avg = dataframe[input_col].mean()
        return dataframe[input_col].fillna(avg)

    return Transformation(
        name="replace-missing-values-with-mean({})".format(column),
        input_column_names=[input_col],
        output_column_name=output_col, 
        function=replace)
