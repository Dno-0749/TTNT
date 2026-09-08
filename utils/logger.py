import logging


def setup_logger():
    """
    Cấu hình hệ thống logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger("AI-Camera")