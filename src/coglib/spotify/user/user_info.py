# pip install dnspython
# pip install grpcio
# pip install grpcio-tools
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. useraccount.proto

import grpc
from google.protobuf.json_format import MessageToDict

from coglib.utils.dns_lookup import resolve

from .useraccount_pb2 import USER_ID, USERNAME, AccountIdentifier, GetAccountRequest
from .useraccount_pb2_grpc import UserAccountStub

user_service = "_spotify-useraccount._grpc.services.gew1.spotify.net"

host = resolve(user_service)

"returns dict with {username, user_id }"


def user_ids(uid: str) -> dict[str, str]:
  identifier = (
    AccountIdentifier(email=uid) if "@" in uid else AccountIdentifier(username=uid)
  )
  with grpc.insecure_channel(host) as channel:
    stub = UserAccountStub(channel)
    request = GetAccountRequest(
      identifier=identifier,
      requested_identifiers=[USER_ID, USERNAME],
    )
    response = stub.GetAccount(
      request=request, timeout=30, metadata=[("service", "someservice")]
    )
    udict = MessageToDict(response)["account"]["identifiers"]
    return {"user_name": udict["username"], "user_id": udict["userId"]}


if __name__ == "__main__":
  email = "rajkumar.junk@gmail.com"
  username = "rajkumar.junk"
  # print(user_ids(username))
  print(user_ids(email))
