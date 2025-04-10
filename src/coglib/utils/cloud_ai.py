import logging
from typing import Any, Dict

import google.cloud.aiplatform as aip

projs = {
  "us": {
    "project_name": "ml-sketchbook",
    "bucket": "gs://cogs_us_tmp/tmp/",
    "sa": "ml-sketchbook@appspot.gserviceaccount.com",
    "location": "us-central1",
  },
  "eu": {
    "project_name": "ml-sketchbook",
    "bucket": "gs://cogs_temp/tmp_eu/",
    "sa": "ml-sketchbook@appspot.gserviceaccount.com",
    "location": "europe-west4",
  },
  "ap": {
    "project_name": "ml-sketchbook",
    "bucket": "gs://cogs_temp/tmp_ap/",
    "sa": "ml-sketchbook@appspot.gserviceaccount.com",
    "location": "asia-east1",
  },
}

acels = {
  "tpu": {
    "vm": "cloud-tpu",
    "acc": aip.gapic.AcceleratorType.TPU_V2.name,
    "n": 8,
    "image": "gcr.io/ml-sketchbook/coglib/tpu:latest",
    "module": "coglib.tf.train",
  },
  "gpu": {
    "vm": "g2-standard-4",
    "acc": aip.gapic.AcceleratorType.NVIDIA_L4.name,
    "n": 1,
    "image": "gcr.io/ml-sketchbook/coglib/gpu_torch:latest",
    "module": "coglib.torch.ddp_train",
  },
  "gpus": {
    "vm": "g2-standard-48",
    "acc": aip.gapic.AcceleratorType.NVIDIA_L4.name,
    "n": 4,
    "image": "gcr.io/ml-sketchbook/coglib/gpu_torch:latest",
    "module": "coglib.torch.ddp_train",
  },
  "hf": {
    "vm": "g2-standard-48",
    "acc": aip.gapic.AcceleratorType.NVIDIA_L4.name,
    "n": 4,
    "image": "gcr.io/ml-sketchbook/coglib/gpu_torch:latest",
    "module": "coglib.hf.train",
  },
}


# proj = projs["us"]
proj = projs["eu"]


def init():
  aip.init(
    project=proj["project_name"],
    service_account=proj["sa"],
    staging_bucket=proj["bucket"],
    location=proj["location"],
  )


# runs a module with given params as a vertex-ai job
def create_vai_job(params: Dict[str, Any] = {}):
  name = params.get("name", "por_en_test")
  data = params.get("data", "ml-sketchbook.cogs_data.por_en")
  epochs = params.get("epochs", 3)
  uri_prep = params.get("uri_prep", True)
  version = params.get("version", "gpus")
  acc = acels[version]
  machine = acc["acc"]
  module = acc["module"]
  image = acc["image"]
  os = f"\n{version}:{module}:{machine}:{image}\n{name}:{data}\n"
  sync = params.get("debug", False)
  print(os, flush=True)
  logging.info(os)

  init()
  vai_job = aip.CustomContainerTrainingJob(
    display_name=acc["vm"] + "_" + str(acc["n"]),
    command=["python3"],
    container_uri=acc["image"],
    model_serving_container_image_uri=acc["image"],
    project=proj["project_name"],
  )
  model_dir = proj["bucket"]

  # args to pass to python
  CMDARGS = ["-m", module, name, data, epochs, uri_prep]
  logging.info(f"Starting job {module} ({name}, {data}, {epochs}) ")
  model = vai_job.run(
    args=CMDARGS,
    replica_count=1,
    model_display_name="cogs-model",
    machine_type=acc["vm"],
    accelerator_type=acc["acc"],
    accelerator_count=acc["n"],
    base_output_dir=model_dir,
    sync=sync,
    # sync=True,  # important for debugging
  )
  return vai_job, model


def get_status(resource_name: str) -> str:
  init()
  job = aip.CustomContainerTrainingJob("", "").get(resource_name)
  print(f"Job id: {resource_name} state :: {job.state}")
  return job.state


def train_model(name: str, data: str, epochs: int, version: str = "gpus", debug=False):
  d = {
    "name": name,
    "data": data,
    "epochs": epochs,
    "version": version,
    "debug": debug,
  }
  logging.info(d)
  try:
    print("Starting job")
    j, m = create_vai_job(d)
    return j, m
  except Exception as err:
    logging.error(f"Cogs Vai: Unexpected {err=}, {type(err)=}")
    return None, None


# To debug set sync = True
def test():
  name = "playlist_5m"
  data = "ml-sketchbook.cogs_data.playlist_isrc_5m"
  j, m = train_model(name, data, 3, "tpu", debug=True)
  # j, m = train_model("playlist_2m", data, 1, "gpus")
  # j, m = train_model("playlist_2m", data, 1, "hf")
  print("\n --------- \n")
  print("Job", j, "Model", m)
  print("\n --------- \n")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  test()
