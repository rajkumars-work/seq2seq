import grpc
import sims_pb2
import sims_pb2_grpc

# User grpc to get vectors from the Sims vector service

Audience = "http://vector-service"
SA = "flux-dev-caller@vector-serving-systems.iam.gserviceaccount.com"
Sims_Service = "_spotify-vector-service._grpc.services.gew1.spotify.net"


# returns host:port
def resolve(service=Sims_Service):
  import dns.resolver

  records = dns.resolver.resolve(service, "SRV")
  if len(records) < 1:
    return ""
  return records[0].target.to_text().rstrip(".") + ":" + str(records[0].port)


def sa_access_token():
  import google.auth.transport.requests
  from google.oauth2 import service_account

  SCOPES = ["https://www.googleapis.com/auth/sqlservice.admin"]
  SERVICE_ACCOUNT_FILE = "/Users/rkumar/.ssh/sp-factoids-f4ccda6b82a2.json"
  creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
  )
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  return creds.token


def get_access_token():
  import google.auth
  import google.auth.transport.requests

  scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  creds, project = google.auth.default(scopes=scopes)
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  return creds.token


def get_impersonation_token(sa: str = SA, audience: str = Audience):
  import requests

  # token = sa_access_token(); print(token, len(token))
  token = get_access_token()

  url = f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa}:generateIdToken"
  payload = {"audience": audience, "includeEmail": "true"}
  header = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Bearer {token}",
  }
  response = requests.post(url, data=payload, headers=header)
  return response.json().get("token", "")


token = get_impersonation_token()
host = resolve()

request = sims_pb2.GetVectorsRequest(
  model="v4-0",
  uris=[
    "spotify:track:3meBo3B9V66M7y8QDhdytz",
    "spotify:track:58rPuuJooYn8llxG2Is8rL",
  ],
)

metadata = (
  (
    "spotify-service-identity",
    f"{token}",
  ),
)


def run():
  with grpc.insecure_channel(host) as channel:
    stub = sims_pb2_grpc.VectorsStub(channel)
    response, call = stub.GetVectors.with_call(
      request=request, metadata=metadata, timeout=10
    )
  print(response.vectors)


if __name__ == "__main__":
  run()
