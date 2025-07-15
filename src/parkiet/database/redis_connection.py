import os
import redis
import json
import logging

log = logging.getLogger(__name__)


class RedisConnection:
    """Redis connection manager for queue operations."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        db: int | None = None,
        password: str | None = None,
    ):
        """
        Initialize Redis connection.

        Args:
            host: Redis host. If None, uses environment variable REDIS_HOST or defaults to localhost
            port: Redis port. If None, uses environment variable REDIS_PORT or defaults to 6379
            db: Redis database number. If None, uses environment variable REDIS_DB or defaults to 0
            password: Redis password. If None, uses environment variable REDIS_PASSWORD or defaults to empty string
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(port or os.getenv("REDIS_PORT", "6379"))
        self.db = int(db or os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD", "")

        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True,
        )

    def push_job(self, queue_name: str, job_data: dict) -> None:
        """
        Push a job to a Redis queue.

        Args:
            queue_name: Name of the queue
            job_data: Job data as dictionary
        """
        self.redis_client.lpush(queue_name, json.dumps(job_data))
        log.info(f"Pushed job to queue {queue_name}: {job_data}")

    def pop_job(self, queue_name: str, timeout: int = 0) -> dict | None:
        """
        Pop a job from a Redis queue (blocking).

        Args:
            queue_name: Name of the queue
            timeout: Timeout in seconds (0 = block indefinitely)

        Returns:
            Job data as dictionary or None if timeout
        """
        result = self.redis_client.brpop([queue_name], timeout=timeout)
        if result:
            queue, job_data = result
            return json.loads(job_data)
        return None

    def get_queue_length(self, queue_name: str) -> int:
        """
        Get the length of a queue.

        Args:
            queue_name: Name of the queue

        Returns:
            Queue length
        """
        return self.redis_client.llen(queue_name)

    def health_check(self) -> bool:
        """
        Check if Redis is healthy.

        Returns:
            True if Redis is responding, False otherwise
        """
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            log.error(f"Redis health check failed: {e}")
            return False


def get_redis_connection() -> RedisConnection:
    """
    Factory function to get a Redis connection.
    """
    return RedisConnection()
