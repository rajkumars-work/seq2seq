# Vertex AI worker images

TPU_TORCH_TAG = gcr.io/ml-sketchbook/coglib/tpu_torch:latest
GPU_TORCH_TAG = gcr.io/ml-sketchbook/coglib/gpu_torch:latest
GPU_IMAGE_TAG = gcr.io/ml-sketchbook/coglib/gpu:latest
TPU_IMAGE_TAG = gcr.io/ml-sketchbook/coglib/tpu:latest
TPU_12_TAG = gcr.io/ml-sketchbook/coglib/tpu_12:latest
BEAM_IMAGE_TAG = gcr.io/ml-sketchbook/coglib/beam:latest

tox:
	tox

gpu-torch-push:
	docker build . --platform linux/x86_64 --tag $(GPU_TORCH_TAG) -f docker/Dockerfile.gpu_torch
	docker push $(GPU_TORCH_TAG)

tpu-docker-push:
	docker build . --platform linux/x86_64 --tag $(TPU_IMAGE_TAG) -f docker/Dockerfile.tpu
	docker push $(TPU_IMAGE_TAG)

gpu-docker-push:
	docker build . --platform linux/x86_64 --tag $(GPU_IMAGE_TAG) -f docker/Dockerfile.gpu
	docker push $(GPU_IMAGE_TAG)

beam-docker-push:
	docker build . --platform linux/x86_64 --tag $(BEAM_IMAGE_TAG) -f docker/Dockerfile.beam
	docker push $(BEAM_IMAGE_TAG)

vai-test:
	python -m  src.cogs.utils.test_cloud_ai

reqs:
	uv pip compile --extra dev --upgrade --emit-index-url --python-platform linux pyproject.toml --output-file=requirements/requirements-py38-dev.txt

# - --
#   docs
#   gcloud container images list --repository="us-docker.pkg.dev/deeplearning-platform-release/gcr.io"

