# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: usertrackplays.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14usertrackplays.proto\x12\x19spotify.usertrackplays.v0\x1a\x1fgoogle/protobuf/timestamp.proto\"\xd1\x03\n\x15UserTrackPlaysRequest\x12N\n\nquery_type\x18\x01 \x03(\x0e\x32:.spotify.usertrackplays.v0.UserTrackPlaysRequest.QueryType\x12\x1a\n\x12play_context_types\x18\x02 \x03(\t\x12\x15\n\rplay_contexts\x18\x03 \x03(\t\x12\x0c\n\x04uris\x18\x04 \x03(\t\x12\r\n\x05limit\x18\x07 \x01(\x05\x12\x12\n\nmedia_type\x18\x08 \x01(\t\x12\x18\n\x10\x63\x61nonical_tracks\x18\t \x01(\x08\x12\x12\n\nfield_mask\x18\n \x01(\t\x12\"\n\x1aplay_context_partial_match\x18\x0b \x01(\x08\x12\x0f\n\x07user_id\x18\x0c \x01(\t\x12\x11\n\tuser_name\x18\r \x01(\t\x12\x13\n\x0binclude_iir\x18\x0e \x01(\x08\"y\n\tQueryType\x12\x0f\n\x0bTRACK_PLAYS\x10\x00\x12\x0f\n\x0bTRACK_SKIPS\x10\x01\x12\x11\n\rEPISODE_PLAYS\x10\x02\x12\x11\n\rEPISODE_SKIPS\x10\x03\x12\x11\n\rCHAPTER_PLAYS\x10\x04\x12\x11\n\rCHAPTER_SKIPS\x10\x05\"e\n\x19TrackEpisodePlaysAndSkips\x12H\n\x11track_plays_skips\x18\x01 \x01(\x0b\x32-.spotify.usertrackplays.v0.TrackPlaysAndSkips\"\x8c\x01\n\x12TrackPlaysAndSkips\x12:\n\x0btrack_plays\x18\x01 \x01(\x0b\x32%.spotify.usertrackplays.v0.TrackPlays\x12:\n\x0btrack_skips\x18\x02 \x01(\x0b\x32%.spotify.usertrackplays.v0.TrackPlays\"\xa1\x01\n\tTrackPlay\x12\x11\n\ttrack_uri\x18\x01 \x01(\t\x12\x17\n\x0fmain_artist_uri\x18\x02 \x01(\t\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x14\n\x0cplay_context\x18\x07 \x01(\t\x12\x11\n\tms_played\x18\x08 \x01(\x03\x12\x10\n\x08provider\x18\n \x01(\t\"G\n\nTrackPlays\x12\x39\n\x0btrack_plays\x18\x01 \x03(\x0b\x32$.spotify.usertrackplays.v0.TrackPlay2\x88\x01\n\x15UserTrackPlaysService\x12o\n\x03get\x12\x30.spotify.usertrackplays.v0.UserTrackPlaysRequest\x1a\x34.spotify.usertrackplays.v0.TrackEpisodePlaysAndSkips\"\x00\x62\x06proto3')



_USERTRACKPLAYSREQUEST = DESCRIPTOR.message_types_by_name['UserTrackPlaysRequest']
_TRACKEPISODEPLAYSANDSKIPS = DESCRIPTOR.message_types_by_name['TrackEpisodePlaysAndSkips']
_TRACKPLAYSANDSKIPS = DESCRIPTOR.message_types_by_name['TrackPlaysAndSkips']
_TRACKPLAY = DESCRIPTOR.message_types_by_name['TrackPlay']
_TRACKPLAYS = DESCRIPTOR.message_types_by_name['TrackPlays']
_USERTRACKPLAYSREQUEST_QUERYTYPE = _USERTRACKPLAYSREQUEST.enum_types_by_name['QueryType']
UserTrackPlaysRequest = _reflection.GeneratedProtocolMessageType('UserTrackPlaysRequest', (_message.Message,), {
  'DESCRIPTOR' : _USERTRACKPLAYSREQUEST,
  '__module__' : 'usertrackplays_pb2'
  # @@protoc_insertion_point(class_scope:spotify.usertrackplays.v0.UserTrackPlaysRequest)
  })
_sym_db.RegisterMessage(UserTrackPlaysRequest)

TrackEpisodePlaysAndSkips = _reflection.GeneratedProtocolMessageType('TrackEpisodePlaysAndSkips', (_message.Message,), {
  'DESCRIPTOR' : _TRACKEPISODEPLAYSANDSKIPS,
  '__module__' : 'usertrackplays_pb2'
  # @@protoc_insertion_point(class_scope:spotify.usertrackplays.v0.TrackEpisodePlaysAndSkips)
  })
_sym_db.RegisterMessage(TrackEpisodePlaysAndSkips)

TrackPlaysAndSkips = _reflection.GeneratedProtocolMessageType('TrackPlaysAndSkips', (_message.Message,), {
  'DESCRIPTOR' : _TRACKPLAYSANDSKIPS,
  '__module__' : 'usertrackplays_pb2'
  # @@protoc_insertion_point(class_scope:spotify.usertrackplays.v0.TrackPlaysAndSkips)
  })
_sym_db.RegisterMessage(TrackPlaysAndSkips)

TrackPlay = _reflection.GeneratedProtocolMessageType('TrackPlay', (_message.Message,), {
  'DESCRIPTOR' : _TRACKPLAY,
  '__module__' : 'usertrackplays_pb2'
  # @@protoc_insertion_point(class_scope:spotify.usertrackplays.v0.TrackPlay)
  })
_sym_db.RegisterMessage(TrackPlay)

TrackPlays = _reflection.GeneratedProtocolMessageType('TrackPlays', (_message.Message,), {
  'DESCRIPTOR' : _TRACKPLAYS,
  '__module__' : 'usertrackplays_pb2'
  # @@protoc_insertion_point(class_scope:spotify.usertrackplays.v0.TrackPlays)
  })
_sym_db.RegisterMessage(TrackPlays)

_USERTRACKPLAYSSERVICE = DESCRIPTOR.services_by_name['UserTrackPlaysService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _USERTRACKPLAYSREQUEST._serialized_start=85
  _USERTRACKPLAYSREQUEST._serialized_end=550
  _USERTRACKPLAYSREQUEST_QUERYTYPE._serialized_start=429
  _USERTRACKPLAYSREQUEST_QUERYTYPE._serialized_end=550
  _TRACKEPISODEPLAYSANDSKIPS._serialized_start=552
  _TRACKEPISODEPLAYSANDSKIPS._serialized_end=653
  _TRACKPLAYSANDSKIPS._serialized_start=656
  _TRACKPLAYSANDSKIPS._serialized_end=796
  _TRACKPLAY._serialized_start=799
  _TRACKPLAY._serialized_end=960
  _TRACKPLAYS._serialized_start=962
  _TRACKPLAYS._serialized_end=1033
  _USERTRACKPLAYSSERVICE._serialized_start=1036
  _USERTRACKPLAYSSERVICE._serialized_end=1172
# @@protoc_insertion_point(module_scope)
