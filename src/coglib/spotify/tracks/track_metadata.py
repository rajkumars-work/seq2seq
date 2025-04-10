# pip install grpcio
# pip install grpcio-tools
# python -m grpc_tools.protoc -I../proto --python_out=. --grpc_python_out=. useraccount.proto

import grpc

from ...utils.dns_lookup import resolve
from .track_pb2 import (
  BatchedEntityRequest,
  EntityRequest,
  ExtensionKind,
  ExtensionQuery,
  Track,
)
from .track_pb2_grpc import ExtendedMetadataStub

Metadata_Service = "_spotify-extended-metadata._grpc.services.gew4.spotify.net"

host = resolve(Metadata_Service)


def get_isrcs(uris: list[str]) -> dict[str, str]:
  eq = ExtensionQuery(extension_kind=ExtensionKind.TRACK_V4)
  ers = [EntityRequest(entity_uri=uri, query=[eq]) for uri in uris]
  request = BatchedEntityRequest(entity_request=ers)
  with grpc.insecure_channel(host) as channel:
    stub = ExtendedMetadataStub(channel)
    response = stub.GetExtensions(request=request, timeout=30)
    isrcs = {}
    for r in response.extended_metadata:
      for d in r.extension_data:
        euri = d.entity_uri
        ed = d.extension_data
        track_data = ed.value
        t = Track()
        t.ParseFromString(track_data)
        for eid in t.external_id:
          isrcs[euri] = eid.id
    return dict((uri, isrcs.get(uri, "").lower()) for uri in uris)


# - ----------------------------------------------------
def test():
  uris = [
    "spotify:track:7Dd4ONH9UK6P3QTPVlAmIh",
    "spotify:track:3meBo3B9V66M7y8QDhdytz",
    "spotify:track:58rPuuJooYn8llxG2Is8rL",
  ]
  print(uris)
  isrcs = get_isrcs(uris)
  print("ISRCS", isrcs)


if __name__ == "__main__":
  test()
