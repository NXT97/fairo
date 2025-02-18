// Generated by Cap'n Proto compiler, DO NOT EDIT
// source: actionlib_msgs.capnp

#pragma once

#include <capnp/generated-header-support.h>
#include <kj/windows-sanity.h>

#if CAPNP_VERSION != 9001
#error "Version mismatch between generated code and library headers.  You must use the same version of the Cap'n Proto compiler and library."
#endif

#include "std_msgs.capnp.h"

CAPNP_BEGIN_HEADER

namespace capnp {
namespace schemas {

CAPNP_DECLARE_SCHEMA(cf7675d72191adb9);
CAPNP_DECLARE_SCHEMA(bf587c1bf845ce13);
CAPNP_DECLARE_SCHEMA(a8a7e35f1c428561);
CAPNP_DECLARE_SCHEMA(e8e34dfe5bc7a639);
CAPNP_DECLARE_SCHEMA(e663a6e78d60e92b);
CAPNP_DECLARE_SCHEMA(947e7feb6f00a0e7);
CAPNP_DECLARE_SCHEMA(aeb6b6555ddb7397);
CAPNP_DECLARE_SCHEMA(bfa359cbc426cd40);
CAPNP_DECLARE_SCHEMA(b634108a61323866);
CAPNP_DECLARE_SCHEMA(9be85d2c5e047323);
CAPNP_DECLARE_SCHEMA(ffd1e31950148c0b);
CAPNP_DECLARE_SCHEMA(efe39098404c1541);
CAPNP_DECLARE_SCHEMA(b688d53fabdc6ee9);

}  // namespace schemas
}  // namespace capnp

namespace mrp {
namespace actionlib {

struct GoalID {
  GoalID() = delete;

  class Reader;
  class Builder;
  class Pipeline;

  struct _capnpPrivate {
    CAPNP_DECLARE_STRUCT_HEADER(cf7675d72191adb9, 0, 2)
    #if !CAPNP_LITE
    static constexpr ::capnp::_::RawBrandedSchema const* brand() { return &schema->defaultBrand; }
    #endif  // !CAPNP_LITE
  };
};

struct GoalStatus {
  GoalStatus() = delete;

  class Reader;
  class Builder;
  class Pipeline;
  static constexpr  ::uint8_t K_PENDING = 0u;
  static constexpr  ::uint8_t K_ACTIVE = 1u;
  static constexpr  ::uint8_t K_PREEMPTED = 2u;
  static constexpr  ::uint8_t K_SUCCEEDED = 3u;
  static constexpr  ::uint8_t K_ABORTED = 4u;
  static constexpr  ::uint8_t K_REJECTED = 5u;
  static constexpr  ::uint8_t K_PREEMPTING = 6u;
  static constexpr  ::uint8_t K_RECALLING = 7u;
  static constexpr  ::uint8_t K_RECALLED = 8u;
  static constexpr  ::uint8_t K_LOST = 9u;

  struct _capnpPrivate {
    CAPNP_DECLARE_STRUCT_HEADER(bf587c1bf845ce13, 1, 2)
    #if !CAPNP_LITE
    static constexpr ::capnp::_::RawBrandedSchema const* brand() { return &schema->defaultBrand; }
    #endif  // !CAPNP_LITE
  };
};

struct GoalStatusArray {
  GoalStatusArray() = delete;

  class Reader;
  class Builder;
  class Pipeline;

  struct _capnpPrivate {
    CAPNP_DECLARE_STRUCT_HEADER(b688d53fabdc6ee9, 0, 2)
    #if !CAPNP_LITE
    static constexpr ::capnp::_::RawBrandedSchema const* brand() { return &schema->defaultBrand; }
    #endif  // !CAPNP_LITE
  };
};

// =======================================================================================

class GoalID::Reader {
public:
  typedef GoalID Reads;

  Reader() = default;
  inline explicit Reader(::capnp::_::StructReader base): _reader(base) {}

  inline ::capnp::MessageSize totalSize() const {
    return _reader.totalSize().asPublic();
  }

#if !CAPNP_LITE
  inline ::kj::StringTree toString() const {
    return ::capnp::_::structString(_reader, *_capnpPrivate::brand());
  }
#endif  // !CAPNP_LITE

  inline bool hasStamp() const;
  inline  ::mrp::std::Time::Reader getStamp() const;

  inline bool hasId() const;
  inline  ::capnp::Text::Reader getId() const;

private:
  ::capnp::_::StructReader _reader;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::List;
  friend class ::capnp::MessageBuilder;
  friend class ::capnp::Orphanage;
};

class GoalID::Builder {
public:
  typedef GoalID Builds;

  Builder() = delete;  // Deleted to discourage incorrect usage.
                       // You can explicitly initialize to nullptr instead.
  inline Builder(decltype(nullptr)) {}
  inline explicit Builder(::capnp::_::StructBuilder base): _builder(base) {}
  inline operator Reader() const { return Reader(_builder.asReader()); }
  inline Reader asReader() const { return *this; }

  inline ::capnp::MessageSize totalSize() const { return asReader().totalSize(); }
#if !CAPNP_LITE
  inline ::kj::StringTree toString() const { return asReader().toString(); }
#endif  // !CAPNP_LITE

  inline bool hasStamp();
  inline  ::mrp::std::Time::Builder getStamp();
  inline void setStamp( ::mrp::std::Time::Reader value);
  inline  ::mrp::std::Time::Builder initStamp();
  inline void adoptStamp(::capnp::Orphan< ::mrp::std::Time>&& value);
  inline ::capnp::Orphan< ::mrp::std::Time> disownStamp();

  inline bool hasId();
  inline  ::capnp::Text::Builder getId();
  inline void setId( ::capnp::Text::Reader value);
  inline  ::capnp::Text::Builder initId(unsigned int size);
  inline void adoptId(::capnp::Orphan< ::capnp::Text>&& value);
  inline ::capnp::Orphan< ::capnp::Text> disownId();

private:
  ::capnp::_::StructBuilder _builder;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  friend class ::capnp::Orphanage;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
};

#if !CAPNP_LITE
class GoalID::Pipeline {
public:
  typedef GoalID Pipelines;

  inline Pipeline(decltype(nullptr)): _typeless(nullptr) {}
  inline explicit Pipeline(::capnp::AnyPointer::Pipeline&& typeless)
      : _typeless(kj::mv(typeless)) {}

  inline  ::mrp::std::Time::Pipeline getStamp();
private:
  ::capnp::AnyPointer::Pipeline _typeless;
  friend class ::capnp::PipelineHook;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
};
#endif  // !CAPNP_LITE

class GoalStatus::Reader {
public:
  typedef GoalStatus Reads;

  Reader() = default;
  inline explicit Reader(::capnp::_::StructReader base): _reader(base) {}

  inline ::capnp::MessageSize totalSize() const {
    return _reader.totalSize().asPublic();
  }

#if !CAPNP_LITE
  inline ::kj::StringTree toString() const {
    return ::capnp::_::structString(_reader, *_capnpPrivate::brand());
  }
#endif  // !CAPNP_LITE

  inline bool hasGoalId() const;
  inline  ::mrp::actionlib::GoalID::Reader getGoalId() const;

  inline  ::uint8_t getStatus() const;

  inline bool hasText() const;
  inline  ::capnp::Text::Reader getText() const;

private:
  ::capnp::_::StructReader _reader;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::List;
  friend class ::capnp::MessageBuilder;
  friend class ::capnp::Orphanage;
};

class GoalStatus::Builder {
public:
  typedef GoalStatus Builds;

  Builder() = delete;  // Deleted to discourage incorrect usage.
                       // You can explicitly initialize to nullptr instead.
  inline Builder(decltype(nullptr)) {}
  inline explicit Builder(::capnp::_::StructBuilder base): _builder(base) {}
  inline operator Reader() const { return Reader(_builder.asReader()); }
  inline Reader asReader() const { return *this; }

  inline ::capnp::MessageSize totalSize() const { return asReader().totalSize(); }
#if !CAPNP_LITE
  inline ::kj::StringTree toString() const { return asReader().toString(); }
#endif  // !CAPNP_LITE

  inline bool hasGoalId();
  inline  ::mrp::actionlib::GoalID::Builder getGoalId();
  inline void setGoalId( ::mrp::actionlib::GoalID::Reader value);
  inline  ::mrp::actionlib::GoalID::Builder initGoalId();
  inline void adoptGoalId(::capnp::Orphan< ::mrp::actionlib::GoalID>&& value);
  inline ::capnp::Orphan< ::mrp::actionlib::GoalID> disownGoalId();

  inline  ::uint8_t getStatus();
  inline void setStatus( ::uint8_t value);

  inline bool hasText();
  inline  ::capnp::Text::Builder getText();
  inline void setText( ::capnp::Text::Reader value);
  inline  ::capnp::Text::Builder initText(unsigned int size);
  inline void adoptText(::capnp::Orphan< ::capnp::Text>&& value);
  inline ::capnp::Orphan< ::capnp::Text> disownText();

private:
  ::capnp::_::StructBuilder _builder;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  friend class ::capnp::Orphanage;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
};

#if !CAPNP_LITE
class GoalStatus::Pipeline {
public:
  typedef GoalStatus Pipelines;

  inline Pipeline(decltype(nullptr)): _typeless(nullptr) {}
  inline explicit Pipeline(::capnp::AnyPointer::Pipeline&& typeless)
      : _typeless(kj::mv(typeless)) {}

  inline  ::mrp::actionlib::GoalID::Pipeline getGoalId();
private:
  ::capnp::AnyPointer::Pipeline _typeless;
  friend class ::capnp::PipelineHook;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
};
#endif  // !CAPNP_LITE

class GoalStatusArray::Reader {
public:
  typedef GoalStatusArray Reads;

  Reader() = default;
  inline explicit Reader(::capnp::_::StructReader base): _reader(base) {}

  inline ::capnp::MessageSize totalSize() const {
    return _reader.totalSize().asPublic();
  }

#if !CAPNP_LITE
  inline ::kj::StringTree toString() const {
    return ::capnp::_::structString(_reader, *_capnpPrivate::brand());
  }
#endif  // !CAPNP_LITE

  inline bool hasHeader() const;
  inline  ::mrp::std::Header::Reader getHeader() const;

  inline bool hasStatusList() const;
  inline  ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Reader getStatusList() const;

private:
  ::capnp::_::StructReader _reader;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::List;
  friend class ::capnp::MessageBuilder;
  friend class ::capnp::Orphanage;
};

class GoalStatusArray::Builder {
public:
  typedef GoalStatusArray Builds;

  Builder() = delete;  // Deleted to discourage incorrect usage.
                       // You can explicitly initialize to nullptr instead.
  inline Builder(decltype(nullptr)) {}
  inline explicit Builder(::capnp::_::StructBuilder base): _builder(base) {}
  inline operator Reader() const { return Reader(_builder.asReader()); }
  inline Reader asReader() const { return *this; }

  inline ::capnp::MessageSize totalSize() const { return asReader().totalSize(); }
#if !CAPNP_LITE
  inline ::kj::StringTree toString() const { return asReader().toString(); }
#endif  // !CAPNP_LITE

  inline bool hasHeader();
  inline  ::mrp::std::Header::Builder getHeader();
  inline void setHeader( ::mrp::std::Header::Reader value);
  inline  ::mrp::std::Header::Builder initHeader();
  inline void adoptHeader(::capnp::Orphan< ::mrp::std::Header>&& value);
  inline ::capnp::Orphan< ::mrp::std::Header> disownHeader();

  inline bool hasStatusList();
  inline  ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Builder getStatusList();
  inline void setStatusList( ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Reader value);
  inline  ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Builder initStatusList(unsigned int size);
  inline void adoptStatusList(::capnp::Orphan< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>&& value);
  inline ::capnp::Orphan< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>> disownStatusList();

private:
  ::capnp::_::StructBuilder _builder;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  friend class ::capnp::Orphanage;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
};

#if !CAPNP_LITE
class GoalStatusArray::Pipeline {
public:
  typedef GoalStatusArray Pipelines;

  inline Pipeline(decltype(nullptr)): _typeless(nullptr) {}
  inline explicit Pipeline(::capnp::AnyPointer::Pipeline&& typeless)
      : _typeless(kj::mv(typeless)) {}

  inline  ::mrp::std::Header::Pipeline getHeader();
private:
  ::capnp::AnyPointer::Pipeline _typeless;
  friend class ::capnp::PipelineHook;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
};
#endif  // !CAPNP_LITE

// =======================================================================================

inline bool GoalID::Reader::hasStamp() const {
  return !_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline bool GoalID::Builder::hasStamp() {
  return !_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline  ::mrp::std::Time::Reader GoalID::Reader::getStamp() const {
  return ::capnp::_::PointerHelpers< ::mrp::std::Time>::get(_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline  ::mrp::std::Time::Builder GoalID::Builder::getStamp() {
  return ::capnp::_::PointerHelpers< ::mrp::std::Time>::get(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
#if !CAPNP_LITE
inline  ::mrp::std::Time::Pipeline GoalID::Pipeline::getStamp() {
  return  ::mrp::std::Time::Pipeline(_typeless.getPointerField(0));
}
#endif  // !CAPNP_LITE
inline void GoalID::Builder::setStamp( ::mrp::std::Time::Reader value) {
  ::capnp::_::PointerHelpers< ::mrp::std::Time>::set(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), value);
}
inline  ::mrp::std::Time::Builder GoalID::Builder::initStamp() {
  return ::capnp::_::PointerHelpers< ::mrp::std::Time>::init(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline void GoalID::Builder::adoptStamp(
    ::capnp::Orphan< ::mrp::std::Time>&& value) {
  ::capnp::_::PointerHelpers< ::mrp::std::Time>::adopt(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::mrp::std::Time> GoalID::Builder::disownStamp() {
  return ::capnp::_::PointerHelpers< ::mrp::std::Time>::disown(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}

inline bool GoalID::Reader::hasId() const {
  return !_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline bool GoalID::Builder::hasId() {
  return !_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline  ::capnp::Text::Reader GoalID::Reader::getId() const {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::get(_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline  ::capnp::Text::Builder GoalID::Builder::getId() {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::get(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline void GoalID::Builder::setId( ::capnp::Text::Reader value) {
  ::capnp::_::PointerHelpers< ::capnp::Text>::set(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), value);
}
inline  ::capnp::Text::Builder GoalID::Builder::initId(unsigned int size) {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::init(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), size);
}
inline void GoalID::Builder::adoptId(
    ::capnp::Orphan< ::capnp::Text>&& value) {
  ::capnp::_::PointerHelpers< ::capnp::Text>::adopt(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::capnp::Text> GoalID::Builder::disownId() {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::disown(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}

inline bool GoalStatus::Reader::hasGoalId() const {
  return !_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline bool GoalStatus::Builder::hasGoalId() {
  return !_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline  ::mrp::actionlib::GoalID::Reader GoalStatus::Reader::getGoalId() const {
  return ::capnp::_::PointerHelpers< ::mrp::actionlib::GoalID>::get(_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline  ::mrp::actionlib::GoalID::Builder GoalStatus::Builder::getGoalId() {
  return ::capnp::_::PointerHelpers< ::mrp::actionlib::GoalID>::get(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
#if !CAPNP_LITE
inline  ::mrp::actionlib::GoalID::Pipeline GoalStatus::Pipeline::getGoalId() {
  return  ::mrp::actionlib::GoalID::Pipeline(_typeless.getPointerField(0));
}
#endif  // !CAPNP_LITE
inline void GoalStatus::Builder::setGoalId( ::mrp::actionlib::GoalID::Reader value) {
  ::capnp::_::PointerHelpers< ::mrp::actionlib::GoalID>::set(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), value);
}
inline  ::mrp::actionlib::GoalID::Builder GoalStatus::Builder::initGoalId() {
  return ::capnp::_::PointerHelpers< ::mrp::actionlib::GoalID>::init(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline void GoalStatus::Builder::adoptGoalId(
    ::capnp::Orphan< ::mrp::actionlib::GoalID>&& value) {
  ::capnp::_::PointerHelpers< ::mrp::actionlib::GoalID>::adopt(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::mrp::actionlib::GoalID> GoalStatus::Builder::disownGoalId() {
  return ::capnp::_::PointerHelpers< ::mrp::actionlib::GoalID>::disown(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}

inline  ::uint8_t GoalStatus::Reader::getStatus() const {
  return _reader.getDataField< ::uint8_t>(
      ::capnp::bounded<0>() * ::capnp::ELEMENTS);
}

inline  ::uint8_t GoalStatus::Builder::getStatus() {
  return _builder.getDataField< ::uint8_t>(
      ::capnp::bounded<0>() * ::capnp::ELEMENTS);
}
inline void GoalStatus::Builder::setStatus( ::uint8_t value) {
  _builder.setDataField< ::uint8_t>(
      ::capnp::bounded<0>() * ::capnp::ELEMENTS, value);
}

inline bool GoalStatus::Reader::hasText() const {
  return !_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline bool GoalStatus::Builder::hasText() {
  return !_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline  ::capnp::Text::Reader GoalStatus::Reader::getText() const {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::get(_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline  ::capnp::Text::Builder GoalStatus::Builder::getText() {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::get(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline void GoalStatus::Builder::setText( ::capnp::Text::Reader value) {
  ::capnp::_::PointerHelpers< ::capnp::Text>::set(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), value);
}
inline  ::capnp::Text::Builder GoalStatus::Builder::initText(unsigned int size) {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::init(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), size);
}
inline void GoalStatus::Builder::adoptText(
    ::capnp::Orphan< ::capnp::Text>&& value) {
  ::capnp::_::PointerHelpers< ::capnp::Text>::adopt(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::capnp::Text> GoalStatus::Builder::disownText() {
  return ::capnp::_::PointerHelpers< ::capnp::Text>::disown(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}

inline bool GoalStatusArray::Reader::hasHeader() const {
  return !_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline bool GoalStatusArray::Builder::hasHeader() {
  return !_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline  ::mrp::std::Header::Reader GoalStatusArray::Reader::getHeader() const {
  return ::capnp::_::PointerHelpers< ::mrp::std::Header>::get(_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline  ::mrp::std::Header::Builder GoalStatusArray::Builder::getHeader() {
  return ::capnp::_::PointerHelpers< ::mrp::std::Header>::get(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
#if !CAPNP_LITE
inline  ::mrp::std::Header::Pipeline GoalStatusArray::Pipeline::getHeader() {
  return  ::mrp::std::Header::Pipeline(_typeless.getPointerField(0));
}
#endif  // !CAPNP_LITE
inline void GoalStatusArray::Builder::setHeader( ::mrp::std::Header::Reader value) {
  ::capnp::_::PointerHelpers< ::mrp::std::Header>::set(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), value);
}
inline  ::mrp::std::Header::Builder GoalStatusArray::Builder::initHeader() {
  return ::capnp::_::PointerHelpers< ::mrp::std::Header>::init(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline void GoalStatusArray::Builder::adoptHeader(
    ::capnp::Orphan< ::mrp::std::Header>&& value) {
  ::capnp::_::PointerHelpers< ::mrp::std::Header>::adopt(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::mrp::std::Header> GoalStatusArray::Builder::disownHeader() {
  return ::capnp::_::PointerHelpers< ::mrp::std::Header>::disown(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}

inline bool GoalStatusArray::Reader::hasStatusList() const {
  return !_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline bool GoalStatusArray::Builder::hasStatusList() {
  return !_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline  ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Reader GoalStatusArray::Reader::getStatusList() const {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>::get(_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline  ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Builder GoalStatusArray::Builder::getStatusList() {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>::get(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline void GoalStatusArray::Builder::setStatusList( ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Reader value) {
  ::capnp::_::PointerHelpers< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>::set(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), value);
}
inline  ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>::Builder GoalStatusArray::Builder::initStatusList(unsigned int size) {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>::init(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), size);
}
inline void GoalStatusArray::Builder::adoptStatusList(
    ::capnp::Orphan< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>&& value) {
  ::capnp::_::PointerHelpers< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>::adopt(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>> GoalStatusArray::Builder::disownStatusList() {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::mrp::actionlib::GoalStatus,  ::capnp::Kind::STRUCT>>::disown(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}

}  // namespace
}  // namespace

CAPNP_END_HEADER

