import json
import os
import pathlib
from typing import List

import gcsfs
from google.cloud import storage  # type: ignore

"""Utilities to work with Google Cloud Storage """


client = storage.Client()


def gfs():
  cfile = "gs://cogs_ds/creds/cogs.json"
  cdict = json.loads(read_text(cfile))
  return gcsfs.GCSFileSystem(project="ml-sketchbook", token=cdict)


# writes to directory; creates if not exist
def download_prefix(src: str, dest: str, overwrite: bool = False, suffix: str = ""):
  bucket, prefix = bucket_prefix(src)
  pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
  for blob in _list_blobs(prefix, bucket, ending=suffix):
    bfn = _blob_filename(blob)
    if not suffix or (suffix and bfn.endswith(suffix)):
      dest_path = dest + "/" + bfn
      if not pathlib.Path(dest_path).exists() or overwrite:
        blob.download_to_filename(dest_path)


def download_file(src: str, dest: str, overwrite: bool = False):
  path = pathlib.Path(dest)
  fname = src.split("/")[-1]
  dest_path = f"{dest}/{fname}" if path.is_dir() else dest
  if pathlib.Path(dest_path).is_file() and not overwrite:
    return
  bucket, prefix = bucket_prefix(src)
  bref = _get_bref(bucket)
  blob = bref.blob(prefix)
  blob.download_to_filename(dest_path)


def upload_file(file: str, prefix: str, bucket: str) -> None:
  path = f"{prefix}{os.path.basename(file)}" if prefix.endswith("/") else prefix
  blob = _get_bref(bucket).blob(path)
  blob.upload_from_filename(file)


def upload_prefix(file: str, target: str) -> None:
  bucket, prefix = bucket_prefix(target)
  upload_file(file, prefix, bucket)


def upload_dir(dir: str, target: str) -> None:
  bucket, prefix = bucket_prefix(target)
  bucket = client.bucket(bucket)

  path = pathlib.Path(dir)
  paths = path.rglob("*")
  files = [path for path in paths if path.is_file()]
  relative_paths = [path.relative_to(dir) for path in files]
  string_paths = [str(path) for path in relative_paths]
  results = storage.transfer_manager.upload_many_from_filenames(
    bucket, string_paths, source_directory=dir, max_workers=8
  )
  nr = zip(string_paths, results)
  return [n for n, r in nr if not isinstance(r, Exception)]


def _blob_filename(blob):
  return blob.name.split("/")[-1]


def read_text(path) -> str:
  bucket, prefix = bucket_prefix(path)
  bref = _get_bref(bucket)
  blob = bref.blob(prefix)
  return blob.download_as_text()


def get_folders(prefix, ending, bucket):
  """get folders that contain file with ending"""
  blobs = _list_blobs(prefix=prefix, ending=ending, bucket=bucket, dirs=True)
  return ["/".join(blob.name.split("/")[:-1]) for blob in blobs]


def get_dirs(path, ending="") -> List[str]:
  bucket, prefix = bucket_prefix(path)
  blobs = _get_bref(bucket).list_blobs(prefix=prefix)
  ending = ending if ending else "/"
  blobs = (b for b in blobs if b.name.endswith(ending))
  last = -2 if ending else -1
  return ["/".join(b.name.split("/")[1:last]) for b in blobs]


def get_files(path, ending) -> List[str]:
  bucket, prefix = bucket_prefix(path)
  blobs = _list_blobs(prefix=prefix, ending=ending, bucket=bucket)
  return [blob.name.split("/")[-1] for blob in blobs]


def get_folder(path, ending):
  """get a folder that contain file with ending"""
  bucket, prefix = bucket_prefix(path)
  if len(bucket) == 0 or len(prefix) == 0:
    return ""
  paths = sorted(
    [f"gs://{bucket}/{folder}" for folder in get_folders(prefix, ending, bucket)]
  )
  return paths[0] if len(paths) > 0 else ""


def path_ok(path: str) -> bool:
  return path.startswith("gs://")


def exists(path: str):
  bucket, prefix = bucket_prefix(path)
  return storage.Blob(bucket=_get_bref(bucket), name=prefix).exists(client)


def rm_file(path: str):
  bucket, prefix = bucket_prefix(path)
  return storage.Blob(bucket=_get_bref(bucket), name=prefix).delete()


def bucket_prefix(path: str):
  """split path of form gs://bucket/.../file to (bucket,.../file)"""
  if path.find("/") == -1:
    return ("", path)
  tokens = [
    token for token in path.split("/") if len(token) > 0 and not token.startswith("gs")
  ]
  return (tokens[0], "/".join(tokens[1:]))


def _get_bref(bucket):
  return client.bucket(bucket)


def _list_blobs(prefix, bucket, ending="", dirs=False):
  """get blobs under a prefix with ending (if specified)"""
  blobs = _get_bref(bucket).list_blobs(prefix=prefix)
  blobs = blobs if dirs else (blob for blob in blobs if not blob.name.endswith("/"))
  blobs = (
    (blob for blob in blobs if blob.name.endswith(ending)) if ending != "" else blobs
  )
  return blobs


def list_blobs(path: str, suffix: str = "", dirs=False):
  bucket, prefix = bucket_prefix(path)
  return _list_blobs(prefix, bucket, suffix)


def list_paths(path: str, suffix: str = "", dirs=False):
  bucket, prefix = bucket_prefix(path)
  return [get_blob_uri(blob) for blob in _list_blobs(prefix, bucket, suffix)]


def download_blob(blob, dest_path: str):
  blob.download_to_filename(dest_path)


def get_blob_uri(blob):
  return "gs://" + blob.id[: -(len(str(blob.generation)) + 1)]
