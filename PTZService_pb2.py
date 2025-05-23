# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: PTZService.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'PTZService.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10PTZService.proto\x12\x06PTZRPC\"\"\n\x0cvoid_request\x12\x12\n\nforrequest\x18\x01 \x01(\x05\"\x1e\n\nvoid_reply\x12\x10\n\x08\x66orreply\x18\x01 \x01(\x05\"\"\n\x08PTZSpeed\x12\n\n\x02hv\x18\x01 \x01(\x05\x12\n\n\x02vv\x18\x02 \x01(\x05\" \n\nPTZBearing\x12\x12\n\nbearingval\x18\x01 \x01(\x05\"\"\n\x0bPTZPitching\x12\x13\n\x0bpitchingval\x18\x01 \x01(\x05\"\x19\n\x06VLZoom\x12\x0f\n\x07zoomval\x18\x01 \x01(\x05\"\x1b\n\x07VLFocus\x12\x10\n\x08\x66ocusval\x18\x01 \x01(\x05\"2\n\x07PTZPose\x12\x12\n\nbearingval\x18\x01 \x01(\x05\x12\x13\n\x0bpitchingval\x18\x02 \x01(\x05\"\x16\n\x07Version\x12\x0b\n\x03ver\x18\x01 \x01(\x05\"\x1e\n\tisInplace\x12\x11\n\tisinplace\x18\x01 \x01(\x08\x32\xff\x11\n\x0cPTZInterface\x12\x32\n\x04init\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x35\n\ngetVersion\x12\x14.PTZRPC.void_request\x1a\x0f.PTZRPC.Version\"\x00\x12;\n\rgetBearingVal\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.PTZBearing\"\x00\x12=\n\x0egetPitchingVal\x12\x14.PTZRPC.void_request\x1a\x13.PTZRPC.PTZPitching\"\x00\x12=\n\x10isBearingInplace\x12\x14.PTZRPC.void_request\x1a\x11.PTZRPC.isInplace\"\x00\x12>\n\x11isPitchingInplace\x12\x14.PTZRPC.void_request\x1a\x11.PTZRPC.isInplace\"\x00\x12\x32\n\x04stop\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x32\n\x04\x64own\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x30\n\x02up\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x32\n\x04left\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x33\n\x05right\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x34\n\x06upleft\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x35\n\x07upright\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x36\n\x08\x64ownleft\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x37\n\tdownright\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12=\n\x13setPositioningSpeed\x12\x10.PTZRPC.PTZSpeed\x1a\x12.PTZRPC.void_reply\"\x00\x12:\n\x10setCruisingSpeed\x12\x10.PTZRPC.PTZSpeed\x1a\x12.PTZRPC.void_reply\"\x00\x12\x36\n\nsetBearing\x12\x12.PTZRPC.PTZBearing\x1a\x12.PTZRPC.void_reply\"\x00\x12\x38\n\x0bsetPitching\x12\x13.PTZRPC.PTZPitching\x1a\x12.PTZRPC.void_reply\"\x00\x12>\n\x15setBearingandPitching\x12\x0f.PTZRPC.PTZPose\x1a\x12.PTZRPC.void_reply\"\x00\x12\x38\n\nsetWiperOn\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x39\n\x0bsetWiperOff\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12;\n\rsetHeadlampOn\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12<\n\x0esetHeadlampOff\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12=\n\x0fsetInitPosition\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x34\n\x06reboot\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12/\n\x07setZoom\x12\x0e.PTZRPC.VLZoom\x1a\x12.PTZRPC.void_reply\"\x00\x12\x31\n\x08setFocus\x12\x0f.PTZRPC.VLFocus\x1a\x12.PTZRPC.void_reply\"\x00\x12:\n\risZoomInplace\x12\x14.PTZRPC.void_request\x1a\x11.PTZRPC.isInplace\"\x00\x12;\n\x0eisFocusInplace\x12\x14.PTZRPC.void_request\x1a\x11.PTZRPC.isInplace\"\x00\x12\x31\n\x07getZoom\x12\x14.PTZRPC.void_request\x1a\x0e.PTZRPC.VLZoom\"\x00\x12\x33\n\x08getFocus\x12\x14.PTZRPC.void_request\x1a\x0f.PTZRPC.VLFocus\"\x00\x12\x34\n\x06zoomIn\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x35\n\x07zoomOut\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x38\n\nzoomInStep\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x39\n\x0bzoomOutStep\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x35\n\x07\x66ocusIn\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12\x36\n\x08\x66ocusOut\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12@\n\x12setFocusManualMode\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x12>\n\x10setFocusAutoMode\x12\x14.PTZRPC.void_request\x1a\x12.PTZRPC.void_reply\"\x00\x42/\n\x1a\x63om.robot.robotGrpc.PTZRPCB\x06PTZRPCP\x01\xa2\x02\x06PTZRPCb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'PTZService_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\032com.robot.robotGrpc.PTZRPCB\006PTZRPCP\001\242\002\006PTZRPC'
  _globals['_VOID_REQUEST']._serialized_start=28
  _globals['_VOID_REQUEST']._serialized_end=62
  _globals['_VOID_REPLY']._serialized_start=64
  _globals['_VOID_REPLY']._serialized_end=94
  _globals['_PTZSPEED']._serialized_start=96
  _globals['_PTZSPEED']._serialized_end=130
  _globals['_PTZBEARING']._serialized_start=132
  _globals['_PTZBEARING']._serialized_end=164
  _globals['_PTZPITCHING']._serialized_start=166
  _globals['_PTZPITCHING']._serialized_end=200
  _globals['_VLZOOM']._serialized_start=202
  _globals['_VLZOOM']._serialized_end=227
  _globals['_VLFOCUS']._serialized_start=229
  _globals['_VLFOCUS']._serialized_end=256
  _globals['_PTZPOSE']._serialized_start=258
  _globals['_PTZPOSE']._serialized_end=308
  _globals['_VERSION']._serialized_start=310
  _globals['_VERSION']._serialized_end=332
  _globals['_ISINPLACE']._serialized_start=334
  _globals['_ISINPLACE']._serialized_end=364
  _globals['_PTZINTERFACE']._serialized_start=367
  _globals['_PTZINTERFACE']._serialized_end=2670
# @@protoc_insertion_point(module_scope)
