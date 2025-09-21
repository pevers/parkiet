import logging
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class DistributedTensorBoardLogger:
    """TensorBoard logger that only writes from the primary process."""

    def __init__(self, log_dir: str, process_id: int):
        self.process_id = process_id
        self.is_primary = process_id == 0
        self.writer = None

        if self.is_primary:
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.is_primary and self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars under a main tag."""
        if self.is_primary and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram of values."""
        if self.is_primary and self.writer:
            self.writer.add_histogram(tag, values, step)

    def flush(self):
        """Flush the writer."""
        if self.is_primary and self.writer:
            self.writer.flush()

    def close(self):
        """Close the writer."""
        if self.is_primary and self.writer:
            self.writer.close()
