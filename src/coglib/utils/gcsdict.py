from pathlib import Path

from sqlitedict import SqliteDict

from .gcs import download_file, exists, upload_prefix


# gcs-storage backed dictionary
class GcsDict:
  def __init__(self, gcs_path: str, create: bool = False):
    self.gcs_path = gcs_path + ".db"
    self.local_file = "/tmp/" + gcs_path.split("/")[-1] + ".db"
    if exists(gcs_path) and not Path(self.local_file).exists():
      download_file(gcs_path, self.local_file)
    flag = "c" if (create or not Path(self.local_file).exists()) else "r"
    self.db = SqliteDict(
      self.local_file, flag=flag, journal_mode="OFF", outer_stack=False
    )

  def __del__(self):
    self.save()

  def commit(self):
    self.db.commit()

  def save(self):
    self.db.commit()
    self.db.close()
    upload_prefix(self.local_file, self.gcs_path)

  def dict(self):
    return self.db
