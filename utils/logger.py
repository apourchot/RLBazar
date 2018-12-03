import pandas as pd 

class Logger(object):

    def __init__(self, path, fields, args=None):
        super(Logger, self).__init__()
        self.path = path
        self.fields = fields
        self.data = pd.DataFrame(columns=fields)

    def save(self):
        """
        Saves the content of the log
        """
        self.data.to_pickle(self.path + "/log.pkl")

    def append(self, values):
        """
        Append value to the log
        """
        res = {self.fields[i]:values[i] for i in range(len(values))}
        self.data = self.data.append(res, ignore_index=True)
        self.save()


