import os
from google.cloud import storage
from pathlib import Path
import logging

log = logging.getLogger(__name__)


class GCSClient:
    """Client for uploading files to Google Cloud Storage."""

    def __init__(
        self,
        bucket_name: str | None = None,
        project_id: str | None = None,
        credentials_path: str | None = None,
    ):
        """
        Initialize GCS client.

        Args:
            bucket_name: GCS bucket name
            project_id: Google Cloud project ID
            credentials_path: Path to service account credentials JSON file
        """
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME", "parkiet-data")
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")

        # Set up credentials if provided
        if credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            creds_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

        # Initialize GCS client
        if self.project_id:
            self.client = storage.Client(project=self.project_id)
        else:
            self.client = storage.Client()

        self.bucket = self.client.bucket(self.bucket_name)

    def upload_file(self, local_file_path: Path, gcs_path: str) -> bool:
        """
        Upload a file to GCS.

        Args:
            local_file_path: Path to local file
            gcs_path: GCS path for the uploaded file

        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_file_path))
            log.info(
                f"Uploaded {local_file_path} to gs://{self.bucket_name}/{gcs_path}"
            )
            return True
        except Exception as e:
            log.error(f"Failed to upload {local_file_path} to GCS: {e}")
            return False

    def upload_chunk(self, local_chunk_path: Path, chunk_filename: str) -> bool:
        """
        Upload a chunk file to the chunks/ directory in GCS.

        Args:
            local_chunk_path: Path to local chunk file
            chunk_filename: Filename of the chunk

        Returns:
            True if successful, False otherwise
        """
        gcs_path = f"chunks/{chunk_filename}"
        return self.upload_file(local_chunk_path, gcs_path)

    def file_exists(self, gcs_path: str) -> bool:
        """
        Check if a file exists in GCS.

        Args:
            gcs_path: GCS path to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except Exception:
            return False

    def get_file_url(self, gcs_path: str, expires_in: int = 3600) -> str | None:
        """
        Generate a signed URL for a file in GCS.

        Args:
            gcs_path: GCS path to the file
            expires_in: URL expiration time in seconds

        Returns:
            Signed URL or None if failed
        """
        try:
            blob = self.bucket.blob(gcs_path)
            url = blob.generate_signed_url(expiration=expires_in)
            return url
        except Exception as e:
            log.error(f"Failed to generate signed URL for {gcs_path}: {e}")
            return None


def get_gcs_client() -> GCSClient:
    """
    Factory function to get a GCS client.
    """
    return GCSClient()
