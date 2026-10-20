class ValidationFailure(Exception):
    """A file failed validation. Becomes REJECTED + MOD_008 with this flag."""

    def __init__(self, flag: str):
        super().__init__(flag)
        self.flag = flag
