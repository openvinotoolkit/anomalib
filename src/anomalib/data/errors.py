class MisMatchError(Exception):
    """Exception raised when a mismatch is detected.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message=""):
        if message:
            self.message = message
        else:
            self.message = "Mismatch detected."
        super().__init__(self.message)
