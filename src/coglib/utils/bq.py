""" "Utility to access bigquery-tables"""

import itertools
import json
import logging
import multiprocessing
import pathlib
import re

from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient, types
from more_itertools import chunked

cpus = multiprocessing.cpu_count()
project = "ml-sketchbook"
service_account = "ml-sketchbook@appspot.gserviceaccount.com"

rclient = BigQueryReadClient()
client = bigquery.Client(project)

Fields = ["context", "target"]
Condition = "target is not null and length(target) > 0"

bqre = r"(bq://)?([\w-]+)(:|.)([\w-]+)(/|.)([\w-]+)"


# Max rows per file when downloading BQ tables
# DEF_MAX_ROWS = 1000000
DEF_MAX_ROWS = 10000


def from_path(path):
  _, project, _, dataset, _, tname = re.match(bqre, path).groups()
  return project, dataset, tname


# - bigquery read-client session helpers ---
# needs fastavro
def _read_session(project, dataset, tname):
  session = types.ReadSession()
  session.table = f"projects/{project}/datasets/{dataset}/tables/{tname}"
  session.data_format = types.DataFormat.AVRO
  session.read_options.selected_fields = Fields
  session.read_options.row_restriction = Condition
  rs = rclient.create_read_session(
    parent=f"projects/{project}", read_session=session, max_stream_count=cpus
  )
  return rs


# - api -----
#
def query_rows(q: str):
  return (dict(r.items()) for r in client.query(q).result())


def table_rows(t: str):
  q = f"select * from `{t}`"
  return query_rows(q)


def get_rows(project, dataset, tname):
  """returns rows of the specified table as an iterable"""
  rs = _read_session(project, dataset, tname)
  readers = [rclient.read_rows(s.name) for s in rs.streams]
  rrows = (r.rows(rs) for r in readers)
  return itertools.chain.from_iterable(rrows)


def path_ok(path):
  return re.match(bqre, path)


def exists(path):
  project, dataset, tname = from_path(path)
  try:
    _ = _read_session(project, dataset, tname)
    return True
  except Exception:
    return False


def read_table(path):
  project, dataset, tname = from_path(path)
  return get_rows(project, dataset, tname)


# if file exists do *not* download
def download_as_json(path, file):
  if not pathlib.Path(file).is_file():
    with open(file, "w") as fh:
      # for d in read_table(path):
      for d in table_rows(path):
        fh.write(json.dumps(d) + "\n")


def download_as_jsons(path, dir, file_prefix="raw", max_rows=DEF_MAX_ROWS):
  if (
    not pathlib.Path(dir).is_dir()
    or pathlib.Path(f"{dir}/{file_prefix}_0.jsons").is_file()
  ):
    return
  for i, c in enumerate(chunked(table_rows(path), max_rows)):
    fn = f"{dir}/{file_prefix}_{i}.jsons"
    if not pathlib.Path(fn).is_file():
      with open(fn, "w") as fh:
        for r in c:
          fh.write(json.dumps(r) + "\n")


# path is project.dataset.table
def upload_from_json(path, file, wait=True):
  """wrties content of file to table. If wait, returns only when job is complete
  Note: Creates client for each call to help parallelism"""
  job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, autodetect=True
  )
  project, dataset, tname = from_path(path)
  table_id = f"{project}.{dataset}.{tname}"
  logging.info(f"Writing table: {table_id}")
  with open(file, "rb") as fh:
    job = bigquery.Client(project=project).load_table_from_file(
      fh, table_id, job_config=job_config
    )
  if wait:
    print(job.result())


if __name__ == "__main__":
  t = "ml-sketchbook.cogs_tmp.learning_test"
  download_as_jsons(t, "/tmp/foo")
