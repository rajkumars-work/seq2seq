# uv pip install dnspython
# Note: naming this modeule dns.py causes problems
import dns.resolver

Vector_Service = "_spotify-vector-service._grpc.services.gew1.spotify.net"
Metadata_Service = "_spotify-extended-metadata._grpc.services.gew1.spotify.net"


# returns host:port
def resolve(service=Vector_Service):
  records = dns.resolver.resolve(service, "SRV")
  if len(records) < 1:
    return ""
  return records[0].target.to_text().rstrip(".") + ":" + str(records[0].port)


if __name__ == "__main__":
  print(resolve(Metadata_Service))
