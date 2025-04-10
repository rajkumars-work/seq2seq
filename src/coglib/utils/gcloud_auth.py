import os

import google.auth
import google.auth.transport.requests
import requests
from google.oauth2 import service_account

from .gcs import download_file

SA_Creds_File = os.environ["HOME"] + "sp-factoids-f4ccda6b82a2.json"
SA_Creds_Path = "gs://cogs_ds/creds/sp-factoids-f4ccda6b82a2.json"
ADC_Path = "gs://cogs_ds/creds/application_default_credentials.json"
MLS_Path = "gs://cogs_ds/creds/application_default_credentials.json"

Audience = "http://vector-service"
SA = "flux-dev-caller@vector-serving-systems.iam.gserviceaccount.com"
Sims_Service = "_spotify-vector-service._grpc.services.gew1.spotify.net"
Scopes = [
  "https://www.googleapis.com/auth/sqlservice.admin",
  "https://www.googleapis.com/auth/cloud-platform",
]


def sa_access_token(sa_file: str):
  return service_account.Credentials.from_service_account_file(sa_file, scopes=Scopes)


def default_creds():
  creds, _ = google.auth.default(scopes=Scopes)
  return creds


def default_creds_from_gcs(path: str = ADC_Path):
  tempfile = "/tmp/adc.json"
  download_file(path, tempfile, overwrite=True)
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tempfile
  creds, _ = google.auth.default(scopes=Scopes)
  return creds


def get_token(creds) -> str:
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  return creds.token


def get_impersonation_token(auth_token: str, sa: str = SA, audience: str = Audience):
  url = f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:generateIdToken"
  payload = {"audience": audience, "includeEmail": "true"}
  header = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Bearer {auth_token}",
  }
  response = requests.post(url, data=payload, headers=header)
  return response.json().get("token", "")
