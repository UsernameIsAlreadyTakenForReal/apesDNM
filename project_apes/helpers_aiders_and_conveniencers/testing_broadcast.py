from logger import Logger

logger = Logger()


def test_singleton(message):
    logger.info("testing_broadcast", message)
