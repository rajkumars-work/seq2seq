# pip install grpcio
# pip install grpcio-tools
# python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. usertrackplays.proto

import grpc
from google.protobuf.json_format import MessageToDict

from coglib.utils.dns_lookup import resolve
from coglib.utils.gcloud_auth import (
  default_creds_from_gcs,
  get_impersonation_token,
  get_token,
)

from .user_info import user_ids
from .usertrackplays_pb2 import UserTrackPlaysRequest
from .usertrackplays_pb2_grpc import UserTrackPlaysServiceStub

Audience = "http://user-track-plays"
TP_SA = "spotify-dj@gke-accounts.iam.gserviceaccount.com"

track_plays_service = "_spotify-user-track-plays._grpc.services.guc3.spotify.net"

host = resolve(track_plays_service)


def user_streams(user_name, user_id):
  request = UserTrackPlaysRequest(
    user_name=user_name,
    user_id=user_id,
    query_type=[
      UserTrackPlaysRequest.QueryType.TRACK_PLAYS,
      UserTrackPlaysRequest.QueryType.TRACK_SKIPS,
    ],
  )
  with grpc.insecure_channel(host) as channel:
    stub = UserTrackPlaysServiceStub(channel)
    creds = default_creds_from_gcs()
    auth_token = get_token(creds)
    im_token = get_impersonation_token(auth_token, TP_SA, audience=Audience)
    metadata = (("spotify-service-identity", f"{im_token}"),)
    response = stub.get(request=request, timeout=30, metadata=metadata)
    rdict = MessageToDict(response)
    return rdict["trackPlaysSkips"]


def ordered_user_streams(user_name, user_id):
  """ordered by time"""
  streams = user_streams(user_name, user_id)
  completes = streams["trackPlays"]["trackPlays"]
  skips = streams["trackSkips"]["trackPlays"]
  for s in skips:
    s.update({"skip": True})
  for c in completes:
    c.update({"skip": False})
  return sorted(completes + skips, key=lambda x: x["timestamp"])


# - api -----------
def user_tracks(user, skips=False) -> list[str]:
  ud = user_ids(user)
  if skips:
    return [
      s["trackUri"] + " skip" if s["skip"] else s["trackUri"]
      for s in ordered_user_streams(ud["user_name"], ud["user_id"])
    ]
  else:
    return [
      s["trackUri"]
      for s in ordered_user_streams(ud["user_name"], ud["user_id"])
      if not s["skip"]
    ]


def user_artists(user, skips=False) -> list[str]:
  ud = user_ids(user)
  if skips:
    return [
      f"{s['mainArtistUri']} skip" if s["skip"] else s["mainArtistUri"]
      for s in ordered_user_streams(ud["user_name"], ud["user_id"])
    ]
  else:
    return [
      s["mainArtistUri"]
      for s in ordered_user_streams(ud["user_name"], ud["user_id"])
      if not s["skip"]
    ]


def user_isrcs(user):
  from ..tracks.track_metadata import get_isrcs

  tracks = user_tracks(user)
  return [i for u, i in get_isrcs(tracks).items()]


# testing
if __name__ == "__main__":
  email = "rajkumar.junk@gmail.com"
  user_name = "rajkumar.junk"
  print(user_isrcs(user_name))
