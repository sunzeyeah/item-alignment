
import logging


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)
