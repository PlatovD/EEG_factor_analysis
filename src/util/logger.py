import logging


class AppLogger:
    _instance: 'AppLogger' = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance is None:
            return cls._instance
        cls._instance = super().__new__(cls, *args, **kwargs)
        cls._instance._set_up_logger()
        return cls._instance

    def _set_up_logger(self):
        self._logger = logging.Logger("AppLogger")
        self._logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self._logger.addHandler(handler)

    def get_logger(self):
        return self._logger


app_logger = AppLogger().get_logger()
