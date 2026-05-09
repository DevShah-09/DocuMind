import os

class LocalStorageService:
    def __init__(self):
        self.upload_dir = "./local_storage"
        # Ensure the directory exists locally
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    async def upload_file(self, file_content: bytes, file_name: str, content_type: str = None):
        """Saves the file to the local disk."""
        file_path = os.path.join(self.upload_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(file_content)
        return file_path

    async def download_file(self, file_name: str) -> bytes:
        """Reads the file from local disk."""
        file_path = os.path.join(self.upload_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_name} not found in storage")
        with open(file_path, "rb") as f:
            return f.read()

    def get_local_url(self, file_name: str):
        """Returns a local static URL for the frontend to display the file."""
        return f"http://localhost:8000/static/{file_name}"

    def delete_file(self, file_name: str):
        """Deletes the file from local disk."""
        file_path = os.path.join(self.upload_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"[STORAGE] Deleted file: {file_name}")
            return True
        return False
