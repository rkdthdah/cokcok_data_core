class EmptyDataError(Exception):
    def __init__(self, data, message="empty data"):
        self.message = f"{message}, data: {data}"
        super().__init__(self.message)

class InvalidDataError(Exception):
    def __init__(self, data, message="invalid data"):
        self.message = f"{message}, data: {data}"
        super().__init__(self.message)

class EmptyValueError(Exception):
    def __init__(self, value, message="empty value"):
        self.message = f"{message}, data: {value}"
        super().__init__(self.message)

class InvalidValueError(Exception):
    def __init__(self, value, message="invalid value"):
        self.message = f"{message}, data: {value}"
        super().__init__(self.message)

class CutDataError(Exception):
    def __init__(self, data, message="raise in method: cutData or simpleCutData"):
        self.message = f"{message}, res_data: {data}"
        super().__init__(self.message)

class MethodOrderError(Exception):
    def __init__(self, message="method interpret must be called after method analysis"):
        self.message = f"{message}"
        super().__init__(self.message)