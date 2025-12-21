#ifndef __CL_HPP_
#define __CL_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <print>
#include <ranges>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// CONFIGURATION & MACROS
// -----------------------------------------------------------------------------

#ifndef CL_DEBUG_LEVEL
    #define CL_DEBUG_LEVEL 0
#endif

#define CL_DEBUG_L1(fmt, ...) do { if constexpr (CL_DEBUG_LEVEL >= 1) std::println(stderr, "[CL_DBG: L1]: {}", std::format(fmt __VA_OPT__(,) __VA_ARGS__)); } while(0)
#define CL_DEBUG_L2(fmt, ...) do { if constexpr (CL_DEBUG_LEVEL >= 2) std::println(stderr, "[CL_DBG: L2]: {}", std::format(fmt __VA_OPT__(,) __VA_ARGS__)); } while(0)
#define CL_DEBUG_L3(fmt, ...) do { if constexpr (CL_DEBUG_LEVEL >= 3) std::println(stderr, "[CL_DBG: L3]: {}", std::format(fmt __VA_OPT__(,) __VA_ARGS__)); } while(0)

#define CL_ASSERT(expr, fmt, ...) \
    do {\
        if (!(expr)) {\
            std::println(stderr, "{}:{}: [\033[1;31mFATAL\033[0m]: {}",__FILE_NAME__, __LINE__, std::format(fmt __VA_OPT__(,) __VA_ARGS__));\
            std::exit(EXIT_FAILURE);\
        }\
    } while(false)

#define CL_ASSERT_LOC(loc, expr, fmt, ...) \
    do {\
        if (!(expr)) {\
            std::println(stderr, "{}:{}: [\033[1;31mFATAL\033[0m]: {}", loc.file_name(), loc.line(), std::format(fmt __VA_OPT__(,) __VA_ARGS__));\
            std::exit(EXIT_FAILURE);\
        }\
    } while(false)

namespace cl {

// -----------------------------------------------------------------------------
// UTILITIES & CONCEPTS
// -----------------------------------------------------------------------------

template<typename T> struct is_std_array : std::false_type {};
template<typename U, std::size_t N> struct is_std_array<std::array<U, N>> : std::true_type {};
template<typename T> inline constexpr bool is_std_array_v = is_std_array<T>::value;

template<typename T> struct is_std_vector : std::false_type {};
template<typename U, typename A> struct is_std_vector<std::vector<U, A>> : std::true_type {};
template<typename T> inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

template<typename T>
concept Supported_Scalar_C = std::is_same_v<T, bool> || std::integral<T> || std::floating_point<T> || std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>;

// Types we can PARSE (Scalars and Arrays)
template<typename T>
concept Parsable_Type_C = Supported_Scalar_C<T> || (is_std_array_v<T> && Supported_Scalar_C<typename T::value_type>);

// Types we can RETRIEVE via get<> (Scalars, Arrays, and Vectors)
using Num    = long long;
using Fp_Num = double;

template<typename T>
concept Gettable_Scalar_C = std::is_same_v<T, Num> || std::is_same_v<T, Fp_Num> || std::is_same_v<T, std::string> || std::is_same_v<T, bool>;

template<typename T>
concept Gettable_Type_C = Gettable_Scalar_C<T> || 
                         (is_std_vector_v<T> && Gettable_Scalar_C<typename T::value_type>) ||
                         (is_std_array_v<T> && Gettable_Scalar_C<typename T::value_type>);

// -----------------------------------------------------------------------------
// MEMORY MANAGEMENT
// -----------------------------------------------------------------------------

class Arena
{
    struct Block
    {
        std::byte* data;
        std::byte* cur;
        std::byte* end;
    };
    std::size_t block_size_;
    std::vector<Block> blocks_;
    std::vector<std::function<void()>> destructors_;

public:
    explicit Arena(std::size_t block_size = 64 * 1024) : block_size_(block_size) { 
        CL_DEBUG_L3("Arena: Initializing with block size {}", block_size);
        add_block(); 
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    ~Arena()
    {
        CL_DEBUG_L2("Arena: Destructing. Objects: {}, Blocks: {}", destructors_.size(), blocks_.size());
        for (auto it = destructors_.rbegin(); it != destructors_.rend(); ++it) (*it)();
        for (auto& b : blocks_) ::operator delete(b.data);
    }

    [[nodiscard]]
    void* alloc(std::size_t n, std::size_t align = alignof(std::max_align_t))
    {
        Block& b = blocks_.back();
        std::byte* p = align_ptr(b.cur, align);

        if (p + n > b.end)
        {
            CL_DEBUG_L3("Arena: Block full. Allocating new block for request of {} bytes", n);
            add_block(std::max(block_size_, n + align));
            return alloc(n, align); 
        }

        b.cur = p + n;
        return p;
    }

    template<typename T, typename... Args>
    [[nodiscard]]
    T* make(Args&&... args)
    {
        void* mem = alloc(sizeof(T), alignof(T));
        T* obj = ::new (mem) T(std::forward<Args>(args)...);

        if constexpr (!std::is_trivially_destructible_v<T>) {
            destructors_.push_back([obj]() { obj->~T(); });
        }
        return obj;
    }

    [[nodiscard]]
    std::string_view str(std::string_view s)
    {
        if (s.empty()) return {};
        char* mem = static_cast<char*>(alloc(s.size() + 1, alignof(char)));
        std::memcpy(mem, s.data(), s.size());
        mem[s.size()] = '\0';
        return { mem, s.size() };
    }

    [[nodiscard]]
    std::string_view str(const char* s)
    {
        if (!s) return {};
        return str(std::string_view{s});
    }

    [[nodiscard]]
    std::size_t blocks() const noexcept { return blocks_.size(); }

private:
    void add_block(std::size_t size = 0)
    {
        std::size_t sz = size ? size : block_size_;
        std::byte* mem = static_cast<std::byte*>(::operator new(sz));
        blocks_.push_back(Block{ mem, mem, mem + sz });
    }

    static std::byte* align_ptr(std::byte* p, std::size_t align)
    {
        auto ip = reinterpret_cast<std::uintptr_t>(p);
        auto aligned = (ip + align - 1) & ~(align - 1);
        return reinterpret_cast<std::byte*>(aligned);
    }
};

// -----------------------------------------------------------------------------
// OPTIONS & VALIDATORS
// -----------------------------------------------------------------------------

template<typename T>
class Validator
{
public:
    virtual ~Validator() = default;
    virtual auto operator()(const T& val) const -> std::expected<void, std::string> = 0;
    virtual std::string help() const { return ""; }
};

template<typename T>
requires Supported_Scalar_C<T>
class Range : public Validator<T>
{
    T l, h;
    public:
    Range(T l, T h) : l(l), h(h) {}
    auto operator()(const T& val) const -> std::expected<void, std::string> override
    {
        if (l <= val && val <= h) return {};
        return std::unexpected(std::format("Value {} out of range [{}, {}]", val, l, h));
    }
    std::string help() const override { return std::format("Range[{}, {}]", l, h); }
};

inline constexpr uint16_t F_REQUIRED      = 1 << 0; 
inline constexpr uint16_t F_HIDDEN        = 1 << 1; 
inline constexpr uint16_t F_MULTI         = 1 << 2; 
inline constexpr uint16_t F_FLAG          = 1 << 3; 
inline constexpr uint16_t F_EXPLICIT_BOOL = 1 << 4; 

template <typename T>
struct Opt
{
    std::array<std::string, 2> args;
    std::string desc{};
    T default_val_{};
    uint16_t flags_ = 0;
    std::string meta_{};
    std::string env_{};

    std::vector<std::shared_ptr<Validator<T>>> validators_;

    auto flags(uint16_t f) -> Opt<T> &
    {
        if constexpr (std::is_same_v<T, bool>) CL_ASSERT((!(f & F_MULTI)),"{}", "Boolean options cannot be multi");
        flags_ |= f;
        return *this;
    }

    auto required() -> Opt<T>& { return flags(F_REQUIRED); }
    auto multi() -> Opt<T>& { return flags(F_MULTI); }

    auto deflt(T x) -> Opt<T>&
    requires (!is_std_array_v<T>)
    {
        default_val_ = std::move(x);
        return *this;
    }

    template <typename... Args>
    auto deflt(Args &&...args) -> Opt<T> &
    requires(is_std_array_v<T> && sizeof...(Args) <= std::tuple_size_v<T> && (std::convertible_to<Args, typename T::value_type> && ...))
    {
        default_val_ = {};
        std::size_t i = 0;
        ((default_val_[i++] = static_cast<typename T::value_type>(std::forward<Args>(args))), ...);
        return *this;
    }

    auto meta(const std::string &s) -> Opt<T> &
    {
        meta_ = s;
        return *this;
    }

    auto env(const std::string &s) -> Opt<T> &
    {
        env_ = s;
        return *this;
    }

    template <std::derived_from<Validator<T>>... Vals>
    auto validators(Vals &&...vals) -> Opt<T> &
    {
        (validators_.push_back(std::make_shared<Vals>(std::forward<Vals>(vals))), ...);
        return *this;
    }

    auto validators(std::shared_ptr<Validator<T>> v) -> Opt<T> &
    {
        validators_.push_back(v);
        return *this;
    }
};

using Opt_id = uint32_t;

enum Opt_type : uint8_t { Int, Bool, Str, Float };

enum Storage_kind : uint8_t { Scalar, Array, Vector };

// Get Opt_type from Parsable types (used in add())
template<typename T>
requires Parsable_Type_C<T>
[[nodiscard]]
consteval Opt_type parsable_to_opt_type()
{
    if constexpr (is_std_array_v<T>)                         return parsable_to_opt_type<typename T::value_type>();
    if constexpr (std::is_same_v<T, bool>)                   return Opt_type::Bool;
    if constexpr (std::integral<T>)                          return Opt_type::Int;
    if constexpr (std::floating_point<T>)                    return Opt_type::Float;
    if constexpr (std::is_convertible_v<T, std::string>)     return Opt_type::Str;
    if constexpr (std::is_convertible_v<T, std::string_view>)return Opt_type::Str;
}

// Get Opt_type from Gettable types (used in get())
template<typename T>
requires Gettable_Type_C<T>
[[nodiscard]]
consteval Opt_type gettable_to_opt_type()
{
    if constexpr (is_std_array_v<T>)                         return gettable_to_opt_type<typename T::value_type>();
    if constexpr (is_std_vector_v<T>)                        return gettable_to_opt_type<typename T::value_type>();
    if constexpr (std::is_same_v<T, bool>)                   return Opt_type::Bool;
    if constexpr (std::integral<T>)                          return Opt_type::Int;
    if constexpr (std::floating_point<T>)                    return Opt_type::Float;
    if constexpr (std::is_convertible_v<T, std::string>)     return Opt_type::Str;
    if constexpr (std::is_convertible_v<T, std::string_view>)return Opt_type::Str;
}

// Get Storage_kind from any Gettable type
template<typename T>
requires Gettable_Type_C<T>
[[nodiscard]]
consteval Storage_kind type_to_storage_kind()
{
    if constexpr (is_std_vector_v<T>)      return Storage_kind::Vector;
    if constexpr (is_std_array_v<T>)       return Storage_kind::Array;
    return Storage_kind::Scalar;
}

// Map enum to canonical storage types
template<Opt_type T> struct _opt_type_to_canonical_type_t_ {};
template<> struct _opt_type_to_canonical_type_t_<Opt_type::Int>   { using value = long long; };
template<> struct _opt_type_to_canonical_type_t_<Opt_type::Bool>  { using value = bool; };
template<> struct _opt_type_to_canonical_type_t_<Opt_type::Str>   { using value = std::string; }; 
template<> struct _opt_type_to_canonical_type_t_<Opt_type::Float> { using value = double; };

// -----------------------------------------------------------------------------
// PARSER CONFIG & STREAM
// -----------------------------------------------------------------------------

inline constexpr uint16_t P_SPACE = 1 << 0;
inline constexpr uint16_t P_COMMA = 1 << 1;
inline constexpr uint16_t P_EQUAL = 1 << 2;
inline constexpr uint16_t P_MULTI_INSTANCE = 1 << 3;

enum Repeated_scalar_policy : uint8_t { REJECT, FIRST, LAST };

class Arg_stream
{
    std::span<char*> args_;
    size_t cur_ = 1;

public:
    Arg_stream(int argc, char** argv) : args_(argv, static_cast<size_t>(argc)) {}

    [[nodiscard]] bool empty() const { return cur_ >= args_.size(); }

    [[nodiscard]] std::optional<std::string_view> peek() const
    {
        if (empty()) return std::nullopt;
        return args_[cur_];
    }

    std::optional<std::string_view> pop()
    {
        if (empty()) return {};
        return args_[cur_++]; 
    }

    void rewind() { if (cur_ > 1) cur_--; }
};

struct Parser_config
{
    uint8_t value_binding = P_EQUAL | P_SPACE;
    uint8_t multi_style   = P_MULTI_INSTANCE | P_COMMA | P_SPACE;
    uint8_t array_style   = P_SPACE | P_COMMA;

    bool allow_combined_short_flags = true;
    bool allow_short_value_concat   = true;
    bool stop_on_double_dash        = true;

    Repeated_scalar_policy repeated_scalar = Repeated_scalar_policy::REJECT;
    bool allow_empty_arrays = false;
};

// -----------------------------------------------------------------------------
// PARSER
// -----------------------------------------------------------------------------

class Parser
{
public:
    struct Option
    {
        Opt_id id;
        std::array<std::string_view, 2> names;
        std::string_view desc;
        Opt_type type;
        Storage_kind storage;
        uint16_t flags;
        std::size_t arity;
        std::string_view meta;
        std::string_view env;

        void* value;

        std::string default_val_str;
        std::vector<std::string> validator_helps;
    };

private:
    struct Runtime
    {
        std::vector<std::string_view> tokens;
        bool seen = false;
    };

    Arena arena_;
    std::string name_;
    std::string description_;
    Opt_id next_id_ = 0;

    std::unordered_map<std::string_view, Opt_id> long_arg_to_id_;
    std::unordered_map<std::string_view, Opt_id> short_arg_to_id_;
    std::vector<Option *> options_;
    std::vector<Runtime> runtime_;
    std::vector<std::string_view> positional_args_;

    friend class std::formatter<Parser::Option>;

public:
    Parser_config cfg_;

    explicit Parser(std::string s = "", std::string des = "", std::size_t reserve = 15)
    : name_(std::move(s)), description_(std::move(des)), arena_(Arena())
    {
        options_.reserve(reserve);
        runtime_.reserve(reserve);
        long_arg_to_id_.reserve(reserve);
        short_arg_to_id_.reserve(52); 
    }

    template <typename T>
    requires Parsable_Type_C<T>
    auto add(Opt<T> opt, std::source_location loc = std::source_location::current()) -> Opt_id;

    template <typename T>
    requires Gettable_Type_C<T>
    [[nodiscard]]
    auto get(Opt_id id, std::source_location loc = std::source_location::current()) const -> const T&
    {
        CL_ASSERT_LOC(loc, id < options_.size(), "Invalid Option ID: {}", id);
        Option* o = options_[id];

        constexpr Opt_type enum_type = gettable_to_opt_type<T>();
        constexpr Storage_kind storage_kind = type_to_storage_kind<T>();

        // TODO: Fix this
        //CL_ASSERT(o->type == enum_type, "Type mismatch in get<T> for option id {}", id, enum_type, o->type);
        CL_ASSERT_LOC(loc, o->type == enum_type, "Type mismatch in get<T> for option id {}", id);
        CL_ASSERT_LOC(loc, o->storage == storage_kind, "Storage kind mismatch in get<T> for option id {}", id);

        CL_DEBUG_L3("Get: ID {} -> ptr {:p}, Type match confirmed.", id, o->value);
        return *reinterpret_cast<const T*>(o->value);
    }


    template <typename T>
    requires Gettable_Type_C<T>
    [[nodiscard]]
    auto get(std::string_view name, std::source_location loc = std::source_location::current()) const -> const T&
    {
        if (name.size() > 1)
        {
            if (auto it = this->long_arg_to_id_.find(name); it != this->long_arg_to_id_.end()) 
                return this->get<T>(it->second);
            else 
                panic(loc, "Long option name: '{}' not found", name);
        }
        else
        {
            if (auto it = this->short_arg_to_id_.find(name); it != this->short_arg_to_id_.end())
                return this->get<T>(it->second);
            if (auto it = this->long_arg_to_id_.find(name); it != this->long_arg_to_id_.end()) 
                return this->get<T>(it->second);

            panic(loc, "[Short/Long] option name: '{}' not found", name);
        }
    }

    auto parse(int argc, char *argv[]) -> std::expected<void, std::string>;
    auto print_help(std::ostream& os = std::cout) -> void;

    [[nodiscard]]
    auto positionals() const -> const std::vector<std::string_view>& { return positional_args_; }

private:
    template <typename... Args>
    [[noreturn]]
    inline auto panic(std::source_location loc, std::format_string<Args...> fmt, Args&&... args) const -> void
    {
        std::println(stderr, "{}:{}: [ERR]: {}", loc.file_name(), loc.line(), std::format(fmt, std::forward<Args>(args)...));
        std::exit(EXIT_FAILURE);
    }

    inline auto assign_id() -> Opt_id { return this->next_id_++; }

    template <typename Dest>
    static auto string_to_value(std::string_view s) -> std::expected<Dest, std::string>
    {
        if constexpr (std::is_same_v<Dest, std::string> || std::is_same_v<Dest, std::string_view>)
        {
            return std::string(s);
        }
        else if constexpr (std::is_same_v<Dest, bool>)
        {
            if (s == "true" || s == "1" || s == "on" || s == "yes" || s == "y") return true;
            if (s == "false" || s == "0" || s == "off" || s == "no" || s == "n") return false;
            CL_DEBUG_L3("ParseValue: Failed bool conversion for '{}'", s);
            return std::unexpected(std::format("Invalid boolean value: '{}'", s));
        }
        else if constexpr (std::is_integral_v<Dest>)
        {
            Dest v{};
            auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
            if (ec != std::errc{}) {
                CL_DEBUG_L3("ParseValue: Failed int conversion for '{}'", s);
                return std::unexpected(std::format("Invalid integer: '{}'", s));
            }
            return v;
        }
        else if constexpr (std::is_floating_point_v<Dest>)
        {
            Dest v{};
            auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
            if (ec != std::errc{}) {
                CL_DEBUG_L3("ParseValue: Failed float conversion for '{}'", s);
                return std::unexpected(std::format("Invalid float: '{}'", s));
            }
            return v;
        }
    }

    template<typename CanonElem>
    static auto parse_tokens(void* storage, Storage_kind kind, std::size_t arity, const std::vector<std::string_view>& raw_tokens) 
        -> std::expected<void, std::string>
    {
        CL_DEBUG_L3("ParseTokens: Processing {} tokens for storage kind {}", raw_tokens.size(), static_cast<int>(kind));
        
        if (kind == Storage_kind::Vector)
        {
            auto* vec = static_cast<std::vector<CanonElem>*>(storage);
            for (auto sv : raw_tokens)
            {
                auto res = string_to_value<CanonElem>(sv);
                if (!res) return std::unexpected(res.error());
                vec->push_back(*res);
            }
        }
        else if (kind == Storage_kind::Array)
        {
            auto* arr_ptr = static_cast<CanonElem*>(storage);
            if (raw_tokens.size() != arity)
                return std::unexpected(std::format("Expected {} elements, got {}", arity, raw_tokens.size()));

            for (size_t i = 0; i < arity; ++i)
            {
                auto res = string_to_value<CanonElem>(raw_tokens[i]);
                if (!res) return std::unexpected(res.error());
                arr_ptr[i] = *res;
            }
        }
        else // Scalar
        {
            auto* sc = static_cast<CanonElem*>(storage);
            if (raw_tokens.empty())
            {
                if constexpr (std::is_same_v<CanonElem, bool>)
                {
                    *sc = true;
                    return {};
                }
                return std::unexpected("Missing value");
            }
            std::string_view sv = raw_tokens.back();
            auto res = string_to_value<CanonElem>(sv);
            if (!res) return std::unexpected(res.error());
            *sc = *res;
        }
        return {};
    }

    template<typename CanonElem, typename ValidatorVec>
    static auto validate_storage(void* storage, Storage_kind kind, std::size_t arity, const ValidatorVec& validators)
        -> std::expected<void, std::string>
    {
        CL_DEBUG_L3("ValidateStorage: Checking constraints for storage kind {}", static_cast<int>(kind));
        if (validators.empty()) return {};

        if (kind == Storage_kind::Array)
        {
            // Reconstruct array type for validation
            auto* arr_ptr = static_cast<CanonElem*>(storage);
            for (const auto& v : validators)
            {
                // Validator expects the full array, not individual elements
                // This is tricky - we need to know the size at compile time
                // For now, validate element by element
                for (size_t i = 0; i < arity; ++i)
                {
                    auto r = (*v)(arr_ptr[i]);
                    if (!r) return r;
                }
            }
        }
        else if (kind == Storage_kind::Vector)
        {
            auto* vec = static_cast<std::vector<CanonElem>*>(storage);
            for (const auto& item : *vec)
            {
                for (const auto& v : validators)
                {
                    auto r = (*v)(item);
                    if (!r) return r;
                }
            }
        }
        else // Scalar
        {
            auto* val = static_cast<CanonElem*>(storage);
            for (const auto& v : validators)
            {
                auto r = (*v)(*val);
                if (!r) return r;
            }
        }
        return {};
    }
};

} // namespace cl

// -----------------------------------------------------------------------------
// IMPLEMENTATION: FORMATTERS (Global/std Scope)
// -----------------------------------------------------------------------------

template<>
struct std::formatter<cl::Opt_type> : std::formatter<std::string_view>
{
    auto format(cl::Opt_type t, format_context& ctx) const
    {
        std::string_view name = "unknown";
        switch (t) {
            case cl::Opt_type::Int:   name = "Int";   break;
            case cl::Opt_type::Bool:  name = "Bool";  break;
            case cl::Opt_type::Str:   name = "Str";   break;
            case cl::Opt_type::Float: name = "Float"; break;
        }
        return std::formatter<std::string_view>::format(name, ctx);
    }
};

template<>
struct std::formatter<cl::Storage_kind> : std::formatter<std::string_view>
{
    auto format(cl::Storage_kind k, format_context& ctx) const
    {
        std::string_view name = "unknown";
        switch (k) {
            case cl::Storage_kind::Scalar: name = "Scalar"; break;
            case cl::Storage_kind::Array:  name = "Array";  break;
            case cl::Storage_kind::Vector: name = "Vector"; break;
        }
        return std::formatter<std::string_view>::format(name, ctx);
    }
};

template<>
struct std::formatter<cl::Parser::Option, char>
{
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    auto format(const cl::Parser::Option& o, std::format_context& ctx) const
    {
        auto out = ctx.out();
        std::format_to(out, "Option{{ id={}, type={}, storage={}, arity={}, flags=0b{:b} }}", 
                      o.id, o.type, o.storage, o.arity, o.flags);
        return out;
    }
};

namespace cl {

// -----------------------------------------------------------------------------
// IMPLEMENTATION: PARSER::ADD
// -----------------------------------------------------------------------------

template <typename T>
requires cl::Parsable_Type_C<T>
auto cl::Parser::add(cl::Opt<T> opt, std::source_location loc) -> Opt_id
{
    CL_DEBUG_L1("Add: Registering Option [long='{}', short='{}']", opt.args[0], opt.args[1]);

    auto id = assign_id();

    // 1. Register Names
    if (!opt.args[0].empty())
    {
        auto name = arena_.str(opt.args[0]);
        CL_ASSERT_LOC(loc, (name.size() > 1), "Long option '{}' must be > 1 char", name);
        if (long_arg_to_id_.contains(name)) panic(loc, "Duplicate option: --{}", name);
        long_arg_to_id_.emplace(name, id);
        CL_DEBUG_L2("  -> Mapped long name '--{}' to ID {}", name, id);
    }

    if (!opt.args[1].empty())
    {
        auto name = arena_.str(opt.args[1]);
        CL_ASSERT_LOC(loc, (name.size() == 1), "Short option '{}' must be 1 char", name);
        if (short_arg_to_id_.contains(name)) panic(loc, "Duplicate option: -{}", name);
        short_arg_to_id_.emplace(name, id);
        CL_DEBUG_L2("  -> Mapped short name '-{}' to ID {}", name, id);
    }

    // 2. Analyze Types & Determine Canonical Storage
    constexpr Opt_type target_enum = parsable_to_opt_type<T>();
    using CanonElem = typename _opt_type_to_canonical_type_t_<target_enum>::value;
    
    CL_DEBUG_L2("  -> Type Analysis: UserT={}, CanonicalT={}, OptType={}", typeid(T).name(), typeid(CanonElem).name(), target_enum);

    // Check flags
    bool is_multi = opt.flags_ & F_MULTI;
    CL_ASSERT_LOC(loc, (!(is_std_array_v<T> && is_multi)), "Arrays cannot be multi. Use vector (scalar + F_MULTI).");

    // Auto-set flag bit for bools
    if constexpr (std::is_same_v<T, bool>)
    {
        if (!(opt.flags_ & (F_FLAG | F_EXPLICIT_BOOL)))
        {
            CL_DEBUG_L2("  -> Implicitly adding F_FLAG to boolean option");
            opt.flags_ |= F_FLAG;
        }
    }

    std::size_t arity = 1;
    Storage_kind storage_kind = Storage_kind::Scalar;
    
    if constexpr (is_std_array_v<T>) 
    {
        arity = std::tuple_size_v<T>;
        storage_kind = Storage_kind::Array;
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        if (!(opt.flags_ & F_EXPLICIT_BOOL)) arity = 0;
    }
    
    if (is_multi)
        storage_kind = Storage_kind::Vector;
    
    CL_DEBUG_L2("  -> Config: Arity={}, Storage={}, Flags=0b{:b}", arity, storage_kind, opt.flags_);

    // 3. Allocate Storage & Defaults (USING CANONICAL TYPES)
    void *val_ptr = nullptr;
    std::string default_val_str_cache;

    if (storage_kind == Storage_kind::Vector)
    {
        CL_DEBUG_L3("  -> Storage: Allocating std::vector<{}>", typeid(CanonElem).name());
        auto *vec = arena_.make<std::vector<CanonElem>>();
        val_ptr = vec;
        default_val_str_cache = "[]";
    }
    else if constexpr (is_std_array_v<T>)
    {
        constexpr size_t N = std::tuple_size_v<T>;
        CL_DEBUG_L3("  -> Storage: Allocating std::array<{}, {}>", typeid(CanonElem).name(), N);
        auto *arr = arena_.make<std::array<CanonElem, N>>();

        for (size_t i = 0; i < N; ++i) (*arr)[i] = static_cast<CanonElem>(opt.default_val_[i]);

        val_ptr = arr;
        default_val_str_cache = "[Array]";
    }
    else
    {
        CL_DEBUG_L3("  -> Storage: Allocating Scalar {}", typeid(CanonElem).name());
        auto *scalar = arena_.make<CanonElem>();
        *scalar = static_cast<CanonElem>(opt.default_val_);
        val_ptr = scalar;

        if constexpr (std::is_arithmetic_v<T> || std::is_same_v<T, std::string>)
            default_val_str_cache = std::format("{}", opt.default_val_);
        else
            default_val_str_cache = std::format("{}", (opt.default_val_ ? "true" : "false"));
    }

    // 4. Extract Validator Helps
    std::vector<std::string> v_helps;
    for (const auto &v : opt.validators_) v_helps.push_back(v->help());

    Option *o = arena_.make<Option>(Option{
        .id = id,
        .names = {arena_.str(opt.args[0]), arena_.str(opt.args[1])},
        .desc = arena_.str(opt.desc),
        .type = target_enum,
        .storage = storage_kind,
        .flags = opt.flags_,
        .arity = arity,
        .meta = arena_.str(opt.meta_),
        .env = arena_.str(opt.env_),
        .value = val_ptr,
        .default_val_str = default_val_str_cache,
        .validator_helps = v_helps
    });

    options_.push_back(o);
    runtime_.push_back(Runtime{});
    return id;
}

// -----------------------------------------------------------------------------
// IMPLEMENTATION: PARSER::PARSE
// -----------------------------------------------------------------------------

auto cl::Parser::parse(int argc, char *argv[]) -> std::expected<void, std::string>
{
    CL_DEBUG_L1("Parse: Start (argc={})", argc);
    Arg_stream args(argc, argv);
    bool stop_parsing = false;

    // 1. Tokenization Loop
    while (!args.empty())
    {
        auto opt_tok = args.pop();
        if (!opt_tok) break;

        std::string_view tok = *opt_tok;
        CL_DEBUG_L2("ParseLoop: Processing Token '{}'", tok);

        if (stop_parsing)
        {
            CL_DEBUG_L3("  -> Storing as positional (stop_parsing=true)");
            positional_args_.push_back(tok);
            continue;
        }

        if (tok == "--" && cfg_.stop_on_double_dash)
        {
            CL_DEBUG_L2("  -> Found '--', stopping option parsing");
            stop_parsing = true;
            continue;
        }

        // --- Long Option ---
        if (tok.starts_with("--"))
        {
            std::string_view body = tok.substr(2);
            std::string_view key = body;
            std::optional<std::string_view> explicit_val;

            if (auto eq = body.find('='); eq != std::string_view::npos)
            {
                if (!(this->cfg_.value_binding & P_EQUAL)) return std::unexpected("Equals syntax not allowed");
                key = body.substr(0, eq);
                explicit_val = body.substr(eq + 1);
            }

            if (!long_arg_to_id_.contains(key))
                return std::unexpected(std::format("Unknown option: --{}", key));
            auto id = long_arg_to_id_[key];
            auto *opt = options_[id];
            auto &rt = runtime_[id];

            rt.seen = true;
            CL_DEBUG_L2("  -> Matched Long Option: --{} (ID: {})", key, id);

            if (opt->arity == 0)
            {
                if (explicit_val) return std::unexpected(std::format("Flag --{} cannot take value", key));
                CL_DEBUG_L3("     -> Flag detected, pushing 'true'");
                rt.tokens.push_back("true");
                continue;
            }

            if (explicit_val)
            {
                CL_DEBUG_L3("     -> Using explicit value '{}'", *explicit_val);
                if (opt->arity > 1 && (cfg_.array_style & P_COMMA) && explicit_val->find(',') != std::string_view::npos)
                    for (auto split_r : std::views::split(*explicit_val, ','))
                        rt.tokens.push_back(std::string_view(split_r.begin(), split_r.end()));
                else
                    rt.tokens.push_back(*explicit_val);
            }
            else
            {
                size_t needed = opt->arity;
                if (opt->flags & F_MULTI) needed = 1;

                while (needed > 0)
                {
                    auto next = args.peek();
                    if (!next) break;

                    if (next->starts_with("-"))
                    {
                        double d;
                        auto [p, ec] = std::from_chars(next->data(), next->data() + next->size(), d);
                        if (ec != std::errc{}) break;
                    }

                    std::string_view val = *args.pop();
                    CL_DEBUG_L3("     -> Consuming value arg: '{}'", val);

                    if ((cfg_.array_style & P_COMMA) && val.find(',') != std::string_view::npos)
                    {
                        for (auto split_r : std::views::split(val, ','))
                            rt.tokens.push_back(std::string_view(split_r.begin(), split_r.end()));
                        needed--;
                    }
                    else
                    {
                        rt.tokens.push_back(val);
                        needed--;
                    }
                }
            }
            continue;
        }

        // --- Short Option ---
        if (tok.starts_with("-") && tok.size() > 1)
        {
            std::string_view body = tok.substr(1);

            while (!body.empty())
            {
                std::string_view key = body.substr(0, 1);
                if (!short_arg_to_id_.contains(key))
                    return std::unexpected(std::format("Unknown option: -{}", key));

                auto id = short_arg_to_id_[key];
                auto *opt = options_[id];
                auto &rt = runtime_[id];
                rt.seen = true;
                CL_DEBUG_L2("  -> Matched Short Option: -{} (ID: {})", key, id);

                body = body.substr(1);

                if (opt->arity == 0)
                {
                    CL_DEBUG_L3("     -> Flag detected, pushing 'true'");
                    rt.tokens.push_back("true");
                }
                else
                {
                    if (!body.empty() && cfg_.allow_short_value_concat)
                    {
                        CL_DEBUG_L3("     -> Consuming attached short value: '{}'", body);
                        rt.tokens.push_back(body);
                        body = "";
                    }
                    else
                    {
                        size_t needed = (opt->flags & F_MULTI) ? 1 : opt->arity;
                        while (needed > 0)
                        {
                            auto next = args.peek();
                            if (!next || next->starts_with("-")) break;
                            auto v = *args.pop();
                            CL_DEBUG_L3("     -> Consuming short value arg: '{}'", v);
                            rt.tokens.push_back(v);
                            needed--;
                        }
                    }
                }
            }
            continue;
        }

        CL_DEBUG_L3("  -> Storing as positional: '{}'", tok);
        positional_args_.push_back(tok);
    }

    // 2. Finalization (Parsing & Validation)
    CL_DEBUG_L1("Parse: Finalization Phase");
    for (auto *opt : options_)
    {
        auto &rt = runtime_[opt->id];

        if ((opt->flags & F_REQUIRED) && !rt.seen)
            return std::unexpected(std::format("Missing required option: {}", opt->names[0].empty() ? opt->names[1] : opt->names[0]));

        if (rt.seen)
        {
            CL_DEBUG_L2("  -> Finalizing Option ID {}", opt->id);
            if (!(opt->flags & F_MULTI))
            {
                if (opt->arity > 0 && rt.tokens.empty())
                    return std::unexpected(std::format("Option {} requires value", opt->names[0]));
            }

            // Dispatch based on canonical type
            std::expected<void, std::string> parse_res;
            switch (opt->type)
            {
                case Opt_type::Int:
                    parse_res = parse_tokens<long long>(opt->value, opt->storage, opt->arity, rt.tokens);
                    break;
                case Opt_type::Float:
                    parse_res = parse_tokens<double>(opt->value, opt->storage, opt->arity, rt.tokens);
                    break;
                case Opt_type::Bool:
                    parse_res = parse_tokens<bool>(opt->value, opt->storage, opt->arity, rt.tokens);
                    break;
                case Opt_type::Str:
                    parse_res = parse_tokens<std::string>(opt->value, opt->storage, opt->arity, rt.tokens);
                    break;
            }
            
            if (!parse_res) return parse_res;

            // Validation - we need the validators, but they're not stored anymore
            // We need to handle this differently
        }
    }
    CL_DEBUG_L1("Parse: Success");
    return {};
}

// -----------------------------------------------------------------------------
// IMPLEMENTATION: PRINT HELP
// -----------------------------------------------------------------------------

auto cl::Parser::print_help(std::ostream &os) -> void
{
    CL_DEBUG_L1("Help: Generating Usage Info");
    os << "\n\033[1mUsage:\033[0m " << name_ << " [options] [args]\n";
    if (!description_.empty())
        os << description_ << "\n";
    os << "\n\033[1mOptions:\033[0m\n";

    size_t max_width = 0;
    for (const auto *opt : options_)
    {
        if (opt->flags & F_HIDDEN) continue;

        size_t w = 0;
        if (!opt->names[1].empty())                           w += 2 + opt->names[1].size();
        if (!opt->names[1].empty() && !opt->names[0].empty()) w += 2;
        if (!opt->names[0].empty())                           w += 2 + opt->names[0].size();
        if (!opt->meta.empty())                               w += 1 + opt->meta.size();
        if (w > max_width)                                    max_width = w;
    }
    max_width += 4;

    for (const auto *opt : options_)
    {
        if (opt->flags & F_HIDDEN) continue;

        std::string flags_part;
        if (!opt->names[1].empty())                           flags_part += std::format("-{}", opt->names[1]);
        if (!opt->names[1].empty() && !opt->names[0].empty()) flags_part += ", ";
        if (!opt->names[0].empty())                           flags_part += std::format("--{}", opt->names[0]);
        if (!opt->meta.empty())                               flags_part += std::format(" \033[3m{}\033[0m", opt->meta);

        os << "  " << std::left << std::setw(max_width) << flags_part;
        os << opt->desc;

        if (opt->flags & F_REQUIRED)
            os << " \033[1;33m[Required]\033[0m";
        if (!opt->default_val_str.empty() && opt->arity > 0)
            os << " \033[2m[Default: " << opt->default_val_str << "]\033[0m";

        if (!opt->validator_helps.empty())
        {
            os << " \033[36m{";
            bool f = true;
            for (const auto &h : opt->validator_helps)
            {
                if (!f) os << ", ";
                if (!h.empty()) os << h;
                else os << "Check";
                f = false;
            }
            os << "}\033[0m";
        }

        os << "\n";
    }
    os << "\n";
}

} // namespace cl

#endif // !__CL_HPP_
