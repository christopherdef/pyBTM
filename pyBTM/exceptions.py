
class OutOfOrderException(Exception):
    '''
    Raised when operations are attempted out of order
    '''
    def __init__(self, operation, details=''):
        self.message = f'{operation} has not been completed.\n'
        self.message += details
        super().__init__(self.message)


class NotIndexedException(OutOfOrderException):
    '''
    Raised when an operation was attempted which requires indexing before completing
    '''
    def __init__(self, details='btm.index_documents()'):
        self.operation="Indexing"
        self.details = details
        super().__init__(operation=self.operation, details=details)

class NotLearnedException(OutOfOrderException):
    '''
    Raised when an operation was attempted which requires learning before completing
    '''
    def __init__(self, details='btm.learn_topics()'):
        self.operation="Learning"
        self.details = details
        super().__init__(operation=self.operation, details=details)


class NotInferredException(OutOfOrderException):
    '''
    Raised when an operation was attempted which requires inference before completing
    '''
    def __init__(self, details='btm.infer_documents()'):
        self.operation="Inference"
        self.details = details
        super().__init__(operation=self.operation, details=details)