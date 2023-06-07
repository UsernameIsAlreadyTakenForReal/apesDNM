from logger import Logger

logger = Logger()


def test_singleton():
    logger.info("testing_broadcast", "This is a test for the singleton class idea")
