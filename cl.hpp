#ifndef __CL_HPP_
#define __CL_HPP_

#include <sys/types.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <iostream>
#include <iterator>
#include <new>
#include <optional>
#include <print>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <variant>

// -----------------------------------------------------------------------------
//  CONFIGURATION & MACROS
// -----------------------------------------------------------------------------

namespace cl::debug {
[[noreturn]]
inline auto todo(std::string &&s, std::source_location loc = std::source_location::current())
{
    std::println("{}:{}:{}: [TODO]: {}", loc.file_name(), loc.line(), loc.column(), loc.function_name());
    std::println("        {}", s);
    std::exit(EXIT_FAILURE);
}

[[noreturn]]
inline auto todo(std::source_location loc = std::source_location::current())
{
    todo("", loc);
    std::exit(EXIT_FAILURE);
}

}  // namespace cl::debug

namespace cl::asrt {

template <typename... Args>
inline auto t(bool cond, std::format_string<Args...> fmt, Args &&...args) -> void
{
    if (!cond)
    {
        std::println(stderr, "[Assert ERR]: {}", std::format(fmt, std::forward<Args>(args)...));
        std::exit(EXIT_FAILURE);
    }
}

template <typename... Args>
inline auto tloc(std::source_location loc, bool cond, std::format_string<Args...> fmt, Args &&...args) -> void
{
    if (!cond)
    {
        std::println(stderr, "{}:{}: [Assert ERR]: {}", loc.file_name(), loc.line(), std::format(fmt, std::forward<Args>(args)...));
        std::exit(EXIT_FAILURE);
    }
}

template <typename... Args>
inline auto floc(std::source_location loc, bool cond, std::format_string<Args...> fmt, Args &&...args) -> void
{
    if (cond)
    {
        std::println(stderr, "{}:{}: [Assert ERR]: {}", loc.file_name(), loc.line(), std::format(fmt, std::forward<Args>(args)...));
        std::exit(EXIT_FAILURE);
    }
}

template <typename... Args>
inline auto f(bool cond, std::format_string<Args...> fmt, Args &&...args) -> void
{
    if (cond)
    {
        std::println(stderr, "[Assert ERR]: {}", std::format(fmt, std::forward<Args>(args)...));
        std::exit(EXIT_FAILURE);
    }
}

}

// Trying to follow semver
namespace cl::vrsn {

constexpr inline int major = 0;
constexpr inline int minor = 0;
constexpr inline int patch = 0;

namespace detail {

    consteval std::array<char, 16> make_version()
    {
        std::array<char, 16> buf{};
        int pos = 0;

        auto append_int = [&](int v)
        {
            char tmp[10]{};
            int n = 0;

            do {
                tmp[n++] = char('0' + (v % 10));
                v /= 10;
            } while (v);

            while (n--) buf[pos++] = tmp[n];
        };

        append_int(major);
        buf[pos++] = '.';
        append_int(minor);
        buf[pos++] = '.';
        append_int(patch);
        buf[pos] = '\0';

        return buf;
    }

    constexpr auto storage = detail::make_version();
} // namespace detail
constexpr const char* val = detail::storage.data();
consteval const char* get() { return val; }
} // namespace cl::vrsn

namespace cl {
// -----------------------------------------------------------------------------
// UTILITIES & CONCEPTS
// -----------------------------------------------------------------------------
using Num    = long long;
using Fp_Num = double;
using Text   = std::string;
using Flag   = bool;

template<typename T>
concept Supported_Scalar_C = std::is_same_v<T, bool> || std::is_same_v<T, cl::Num>|| std::is_same_v<T, cl::Fp_Num> || std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>;

template<typename T, std::size_t N>
requires Supported_Scalar_C<T>
using Fix_list = std::array<T, N>;

template<typename T>
requires Supported_Scalar_C<T>
using List = std::vector<T>;

template<typename T> struct is_std_array : std::false_type {};
template<typename U, std::size_t N> struct is_std_array<std::array<U, N>> : std::true_type {};
template<typename T> inline constexpr bool is_std_array_v = is_std_array<T>::value;

template<typename T> struct is_std_vector : std::false_type {};
template<typename U, typename A> struct is_std_vector<std::vector<U, A>> : std::true_type {};
template<typename T> inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

// Types we can PARSE (Scalars and Arrays)
template<typename T>
concept Parsable_Type_C = Supported_Scalar_C<T> || (is_std_array_v<T> && Supported_Scalar_C<typename T::value_type>);

template<typename T>
concept Gettable_Scalar_C = std::is_same_v<T, Num> || std::is_same_v<T, Fp_Num> || std::is_same_v<T, std::string> || std::is_same_v<T, bool>;

template<typename T>
concept Gettable_Type_C = Gettable_Scalar_C<T> || (is_std_vector_v<T> && Gettable_Scalar_C<typename T::value_type>) || (is_std_array_v<T> && Gettable_Scalar_C<typename T::value_type>);

template<typename V, typename T>
concept Validator_C = requires(const V& v, const T& val) {
    { v(val) } -> std::convertible_to<std::expected<void, std::string>>;
    { v.help() } -> std::convertible_to<std::string>;
};
// -----------------------------------------------------------------------------
// MEMORY MANAGEMENT
// -----------------------------------------------------------------------------

namespace detail
{
template <typename T>
struct get_inner_type { using type = T; };

// Specialization for vector to extract inner type
template <typename T, typename A>
struct get_inner_type<std::vector<T, A>> { using type = T; };
}

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

    template<typename... Args>
    constexpr auto deflt(Args... values)
    {
        return [=]<typename Opt>(Opt &o) constexpr
        {
            using T = typename Opt::value_type;
            static_assert(is_std_array_v<T>, "Variadic deflt() can only be used with std::array<T, N>");
            constexpr std::size_t N = std::tuple_size_v<T>;
            static_assert(sizeof...(Args) == N, "Number of default values must match array size");
            o.default_val_ = T{static_cast<typename T::value_type>(values)...};
            o.has_default_ = true;
        };
    }

    explicit Arena(std::size_t block_size = 64 * 1024) : block_size_(block_size)
    {
        add_block();
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    ~Arena()
    {
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

        if constexpr (!std::is_trivially_destructible_v<T>)
            destructors_.push_back([obj]() { obj->~T(); });

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

enum class List_type    : uint8_t { Consecutive, Delimited };
enum class Multi_type   : uint8_t { Repeat, Delimited };
enum class Bool_type    : uint8_t { Flag, Explicit };
enum class Binding_type : uint8_t { Equal, Next, Both };

enum class Flags : uint16_t
{
    O_Multi    = 1,
    O_Required = 1 << 1,
    O_Hidden   = 1 << 2,
    O_Env      = 1 << 3,
    O_Default  = 1 << 4,
    O_Explicit_Bool = 1 << 5
};
constexpr auto operator*(Flags f) -> const uint16_t { return static_cast<uint16_t>(f); }
constexpr Flags operator|(Flags a, Flags b)    noexcept { return static_cast<Flags>(std::to_underlying(a) | std::to_underlying(b)); }
constexpr Flags operator&(Flags a, Flags b)    noexcept { return static_cast<Flags>(std::to_underlying(a) & std::to_underlying(b)); }
constexpr Flags operator^(Flags a, Flags b)    noexcept { return static_cast<Flags>(std::to_underlying(a) ^ std::to_underlying(b)); }
constexpr Flags operator~(Flags f)             noexcept { return static_cast<Flags>(~std::to_underlying(f)); }
constexpr Flags &operator|=(Flags &a, Flags b) noexcept { return a = a | b; }
constexpr Flags &operator&=(Flags &a, Flags b) noexcept { return a = a & b; }

struct List_cfg
{
    List_type type;
    std::string delimiter;
};

struct Multi_cfg
{
    Multi_type type;
    std::string delimiter;
};

template <typename T>
struct Opt
{
    using value_type = T;
    std::array<std::string, 2> args{};
    std::string desc{};
    T default_val{};
    std::string meta_{""};
    std::string env_{""};
    uint16_t flags = 0;
    List_cfg list_cfg { List_type::Consecutive, ","};
    Multi_cfg multi_cfg { Multi_type::Repeat, "," };
    struct ValidatorEntry
    {
        std::function<std::expected<void, std::string>(const T&)> func;
        std::string help;
    };
    std::vector<ValidatorEntry> validators_;
    std::source_location loc{};
};

struct Name_config
{
    std::string_view short_name;
    std::string_view long_name;
    std::source_location loc;
};

constexpr auto name(std::string_view s, std::string_view l, std::source_location loc = std::source_location::current())
{
    return Name_config{s, l, loc};
}

constexpr auto desc(std::string_view sv)
{
    return [sv]<typename Opt>(Opt& o) constexpr { o.desc = sv; };
}

constexpr auto required()
{
    return []<typename Opt>(Opt& o) constexpr { o.flags |= *Flags::O_Required; };
}

constexpr auto explicit_bool()
{
    return []<typename Opt>(Opt &o) constexpr
    {
        static_assert(std::is_same_v<typename Opt::value_type, bool>, "Non bools/flags can't have be explicit bools");
        o.flags |= *Flags::O_Explicit_Bool;
    };
}

constexpr auto multi(Multi_type type = Multi_type::Repeat, std::string_view d = ",")
{
    cl::asrt::t((!d.empty()),"Delimiter cannot be empty.");
    return [type, d]<typename Opt>(Opt &o) constexpr
    {
        static_assert(!is_std_array_v<typename Opt::value_type>, "Fixed-size arrays cannot be multi-value. Use vector.");
        o.multi_cfg.type = type;
        o.multi_cfg.delimiter = d;
        o.flags |= *Flags::O_Multi;
    };
}

constexpr auto array(List_type t = List_type::Consecutive, std::string_view d = ",")
{
    cl::asrt::t((!d.empty()), "delimiter cannot be empty.");
    return [t, d]<typename Opt>(Opt &o) constexpr
    {
        static_assert(is_std_array_v<typename Opt::value_type>, "Array configuration can only be done on arrays");
        o.list_cfg.type = t;
        o.list_cfg.delimiter = d;
    };
}

template<typename T>
constexpr auto deflt(T&& val)
{
    return [val = std::forward<T>(val)]<typename Opt>(Opt& o) constexpr { o.default_val = std::move(val); o.flags |= *Flags::O_Default; };
}

template<typename T>
constexpr auto flags(uint16_t f)
{
    return [f]<typename Opt>(Opt& o) constexpr { o.flags |= f; };
}

template<typename... Args>
constexpr auto deflt(Args... values)
{
    return [=]<typename Opt>(Opt &o) constexpr
    {
        using T = typename Opt::value_type;
        static_assert(is_std_array_v<T>, "Variadic deflt() can only be used with std::array<T, N>");
        constexpr std::size_t N = std::tuple_size_v<T>;
        static_assert(sizeof...(Args) == N, "Number of default values must match array size");
        o.default_val = T{static_cast<typename T::value_type>(values)...};
        o.flags |= *Flags::O_Default;
    };
}

template <typename... Vs>
constexpr auto validators(Vs &&...vals)
{
    return [=]<typename Opt>(Opt &o)
    {
        using T = typename Opt::value_type;
        (
            [&]
            {
                static_assert(Validator_C<Vs, T>, "Argument is not a valid Validator, must have operator() -> expected<void, std::string>; and help() -> std::string;");
                o.validators_.push_back({.func = [v = vals](const T &x) { return v(x); }, .help = vals.help()});
            }(),
        ...);
    };
}

constexpr auto env(std::string_view e)
{
    return [e]<typename Opt>(Opt& o) constexpr { o.flags |= (*Flags::O_Env); o.env_ = e; };
}

constexpr auto meta(std::string_view m)
{
    return [m]<typename Opt>(Opt& o) constexpr { o.meta_ = m; };
}

template<typename Config, typename T>
concept Configurer = requires(Config&& c, Opt<T>& opt) {
    { std::forward<Config>(c)(opt) } -> std::convertible_to<void>;
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

enum class Repeated_scalar_policy : uint8_t { REJECT, FIRST, LAST };

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

    std::string_view pop()
    {
        if (empty()) return {};
        return args_[cur_++]; 
    }

    std::size_t size() { return args_.size() - cur_; }

    void rewind() { if (cur_ > 1) cur_--; }
};

struct Parser_config
{
    Binding_type value_binding = Binding_type::Both; // '=' binding not allowed in case of list type consecutive
    bool allow_combined_short_flags = true;
    bool allow_short_value_concat   = true;
    bool stop_on_double_dash        = true;
    Repeated_scalar_policy repeated_scalar = Repeated_scalar_policy::REJECT;
    bool allow_empty_arrays = false;
};

struct Parse_err
{
    struct err
    {
        std::string option{};
        std::string message{};
    };
    std::vector<err> errors{};

    template <typename... Args>
    auto push_err(const std::string& option, std::format_string<Args...> fmt, Args &&...a)
    {
        this->errors.push_back({option, std::format(fmt, std::forward<Args>(a)...)});
    }

    template <typename... Args>
    auto push_err(std::string_view option, std::format_string<Args...> fmt, Args &&...args)
    {
        return this->push_err(std::string(option), fmt, std::forward<Args>(args)...);
    }
private:
};

using Runtime_value = std::variant<
    cl::Text, cl::Num, cl::Flag, cl::Fp_Num,
    std::vector<cl::Text>,
    std::vector<cl::Num>,
    std::vector<cl::Flag>,
    std::vector<cl::Fp_Num>,
    std::monostate
>;

struct Runtime
{
    Runtime_value runtime_value{std::monostate()};
    bool parsed{false};
    std::size_t count{0};

    Runtime& operator=(const Runtime & r) = default;
};

struct Parse_res
{
    struct Option_info
    {
        std::size_t arity;
        std::string_view env;
        Opt_type type;
        Storage_kind storage;
    };

    std::vector<Runtime> runtime_;
    std::vector<Option_info> opt_info_;

    // TODO: below
    // PERF: Maybe we should return a refernce to be copied or something for performance
    template<typename T>
    auto get(Opt_id id) -> T
    {
        return std::get<T>(runtime_[id].runtime_value);
    }
};

// Inside cl.hpp, after Parse_err definition or outside namespace cl
inline std::ostream& operator<<(std::ostream& os, const cl::Parse_err& err) {
    for (const auto& e : err.errors) {
        os << "Error in option '" << e.option << "': " << e.message << "\n";
    }
    return os;
}
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
        List_cfg list_cfg;
        Multi_cfg multi_cfg;
        std::size_t arity;
        std::string_view meta;
        std::string_view env;
        std::vector<std::string_view> validator_helps;
        std::string_view default_hints;
        Runtime_value default_value;

        std::function<std::expected<void, std::string>(const Runtime_value&)> validate;
    };
    std::vector<std::string> truthy_strs = { "y", "true", "yes", "t" };
    std::vector<std::string> falsy_strs =  { "n", "false", "no", "f" };

private:
    struct Parse_ctx
    {
        Arg_stream & args;
        Parse_res& res;
        Parse_err& err;
        const Parser_config &cfg;
        bool stop_flags;
        std::string_view curr_key{};

        Parse_ctx( Arg_stream& a, Parse_err & e, Parse_res& r, Parser_config& c) : args(a), res(r), err(e), cfg(c), stop_flags(false) {}
        Parse_ctx(const Parse_ctx & p) = delete;
        Parse_ctx(const Parse_ctx && p) = delete;
        Parse_ctx operator=(const Parse_ctx & p) = delete;
        Parse_ctx operator=(const Parse_ctx && p) = delete;

        // Helper to push errors easily
        template <typename... Args>
        void error(std::format_string<Args...> fmt, Args &&...a)
        {
            err.push_err(this->curr_key, fmt, std::forward<Args>(a)...);

            while (!args.empty())
            {
                auto t = args.peek();
                // Stop if we see something that looks like a flag
                if (t)
                {
                    if (t->starts_with("--"))
                        return;
                    if (t->starts_with("-") && t->size() == 2 && std::isalpha((*t)[1]))
                        return;
                }
                args.pop();
            }
        }
    };
    static constexpr int sn_index = 0;
    static constexpr int ln_index = 1;

    Arena arena_;
    std::string name_;
    std::string description_;
    Opt_id next_id_ = 0;

    std::unordered_map<std::string_view, Opt_id> long_arg_to_id_;
    std::unordered_map<std::string_view, Opt_id> short_arg_to_id_;
    std::vector<Option *> options_;
    std::vector<Runtime> runtime_;

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

    void add_explicit_bool_strs(const std::vector<std::string> & truthy, const std::vector<std::string> & falsy);

    template<typename T, typename ...Configs>
    requires ((Configurer<Configs, T> && ...) && Parsable_Type_C<T>)
    auto add(Name_config name_cfg, Configs&&... confs) -> Opt_id;

    auto parse(int argc, char *argv[]) -> std::expected<Parse_res, Parse_err>;

    auto print_help(std::ostream& os = std::cout) -> void;
private:

    template <typename T>
    requires Parsable_Type_C<T>
    inline auto add_impl(Opt<T> opt) -> Opt_id;

    template <typename... Args>
    [[noreturn]]
    inline auto panic(std::format_string<Args...> fmt, Args&&... args) const -> void
    {
        std::println(stderr, "[ERR]: {}", std::format(fmt, std::forward<Args>(args)...));
        std::exit(EXIT_FAILURE);
    }

    inline auto assign_id() -> Opt_id { return this->next_id_++; }

    template <typename Dest>
    requires Supported_Scalar_C<Dest>
    auto string_to_value(std::string_view s) -> std::expected<Dest, std::string>;

    void handle_long_token(Parse_ctx& ctx, std::string_view body);
    void handle_short_token(Parse_ctx& ctx, std::string_view body);
    bool add_short_combined(Parse_ctx& ctx, std::string_view body);
    bool acquire_value(Parse_ctx& ctx, Option* opt, std::optional<std::string_view> explicit_val);
    void inject_value(Parse_ctx &ctx, Option *opt, std::span<const std::string_view> raw_values);
    void assign_true(Runtime& rt);
};

} // namespace cl

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
        std::format_to(out, "Option{{ id={}, type={}, storage={}, arity={}, flags=0b{:b}, default_hints: {}}}", 
                       o.id, o.type, o.storage, o.arity, o.flags, o.default_hints);
        return out;
    }
};

template <>
struct std::formatter<cl::Parse_err>
{
    // No custom format specifiers for now
    constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const cl::Parse_err &pe, FormatContext &ctx) const
    {
        auto out = ctx.out();

        for (std::size_t i = 0; i < pe.errors.size(); ++i)
        {
            const auto &e = pe.errors[i];

            out = std::format_to(out, "error: {}", e.message);

            if (!e.option.empty())
                out = std::format_to(out, "\n      option: {}", e.option);

            if (i + 1 < pe.errors.size())
                out = std::format_to(out, "\n");
        }

        return out;
    }
};

#ifdef CL_IMPLEMENTATION
#include <charconv>
#include <ranges>

template<typename T>
constexpr std::string_view type_name()
{
    #if defined(__clang__) || defined(__GNUC__)
        std::string_view p = __PRETTY_FUNCTION__;
        auto start = p.find("T = ") + 4;
        auto end   = p.find(']', start);
        return p.substr(start, end - start);
    #elif defined(_MSC_VER)
        return typeid(T).name();
        //std::string_view p = __FUNCSIG__;
        //auto start = p.find("type_name<") + 10;
        //auto end   = p.find(">(void)");
        //return p.substr(start, end - start);
    #else
        return typeid(T).name();
        // return "unsupported compiler: send patches :)";
    #endif
}

void cl::Parser::add_explicit_bool_strs(const std::vector<std::string> &truthy, const std::vector<std::string> &falsy)
{
    truthy_strs = truthy;
    falsy_strs = falsy;
}

template <typename T, typename... Configs>
requires((cl::Configurer<Configs, T> && ...) && cl::Parsable_Type_C<T>)
auto cl::Parser::add(Name_config name_cfg, Configs &&...confs) -> Opt_id
{
    Opt<T> opt;
    opt.args[0] = name_cfg.short_name;
    opt.args[1] = name_cfg.long_name;
    opt.loc = name_cfg.loc;
    if (opt.args[0].empty() && opt.args[1].empty())
        panic("Option must have at least one name (short or long)");
    (confs(opt), ...);
    return this->add_impl(opt);
}

template <typename T>
requires cl::Parsable_Type_C<T>
inline auto cl::Parser::add_impl(Opt<T> opt) -> Opt_id
{

    auto is_valid_name = [](std::string_view name)
    {
        return (std::ranges::all_of(name, [](char c) { return std::isalnum(c) || c == '-' || c == '_'; }) && (!name.starts_with("-")));
    };

    auto id = assign_id();

    // 1. Register Names
    if (!opt.args[this->ln_index].empty())
    {
        auto name = arena_.str(opt.args[this->ln_index]);
        cl::asrt::t(is_valid_name(name), "Long option '{}' contains invalid characters, can't sart with '-' and can only contain alphanum, '-' & '_'", name);
        cl::asrt::t((name.size() > 1), "Long option '{}' must be > 1 char", name);
        if (long_arg_to_id_.contains(name)) panic("Duplicate option: --{}", name);
        long_arg_to_id_.emplace(name, id);
    }

    if (!opt.args[this->sn_index].empty())
    {
        auto name = arena_.str(opt.args[this->sn_index]);
        cl::asrt::t((name.size() == 1), "Short option '{}' must be 1 char", name);
        cl::asrt::t((std::isalpha(name[0])), "Short option '{}' must be a letter or alphabet.", name);
        if (short_arg_to_id_.contains(name)) panic("Duplicate option: -{}", name);
        short_arg_to_id_.emplace(name, id);
    }
    if (opt.flags & *Flags::O_Env)
    {
        auto is_valid_env = [](std::string_view name)
        {
            if (name.empty() || std::isdigit(name[0]))
                return false;
            return std::ranges::all_of(name, [](char c) { return std::isalnum(c) || c == '_'; });
        };
        if (!is_valid_env(opt.env_))
            panic("Invalid environment variable name: '{}'", opt.env_);
    }

    // 2. Analyze Types & Determine Canonical Storage
    constexpr Opt_type target_enum = parsable_to_opt_type<T>();
    using CanonElem = typename _opt_type_to_canonical_type_t_<target_enum>::value;


    // Check flags
    bool is_multi = opt.flags & *Flags::O_Multi;
    cl::asrt::t((!(is_std_array_v<T> && is_multi)), "Arrays cannot be multi. Use vector (scalar + F_MULTI).");

    std::size_t arity = 1;
    Storage_kind storage_kind = Storage_kind::Scalar;

    if constexpr (is_std_array_v<T>)
    {
        if (opt.list_cfg.type == List_type::Consecutive && cfg_.value_binding == Binding_type::Equal)
                panic("CONSECUTIVE arrays incompatible with Binding_type::Equal. Use Next or Both.\n\t"
                      "Even then, if you use both, this will break at runtime if you enter consecutive member array with Equal binding because Both contains Equal.");

        arity = std::tuple_size_v<T>;
        storage_kind = Storage_kind::Array;
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        if (!(opt.flags & *Flags::O_Explicit_Bool)) arity = 0;
    }

    if (is_multi)
    {
        if (opt.multi_cfg.type == Multi_type::Delimited)
        {
            if (opt.multi_cfg.delimiter.empty())
                panic("DELIMITED multi mode requires non-empty delimiter");
        }
        storage_kind = Storage_kind::Vector;
    }

    Runtime_value val;

    if (storage_kind == Storage_kind::Vector)
    {
        val = std::vector<CanonElem>{};
    }
    else if constexpr (is_std_array_v<T>)
    {
        constexpr size_t N = std::tuple_size_v<T>;
        std::vector<CanonElem> temp_vec(N, {});
        val = std::move(temp_vec);
    }
    else
    {
        val = CanonElem{};
    }

    // 4. Extract Validator Helps
    std::vector<std::string_view> v_helps;
    for (const auto &v : opt.validators_) v_helps.push_back(this->arena_.str(v.help));

    Option *o = arena_.make<Option>(Option{
        .id = id,
        .names = {arena_.str(opt.args[0]), arena_.str(opt.args[1])},
        .desc = arena_.str(opt.desc),
        .type = target_enum,
        .storage = storage_kind,
        .flags = opt.flags,
        .list_cfg = opt.list_cfg,
        .multi_cfg = opt.multi_cfg,
        .arity = arity,
        .meta = arena_.str(opt.meta_),
        .env = arena_.str(opt.env_),
        .validator_helps = v_helps,
        .default_value = val
    });

    if (opt.flags & (*Flags::O_Default))
    {
        if constexpr (is_std_array_v<T>) {
            std::vector<CanonElem> val;
            for (auto & a : opt.default_val)
                val.push_back(a);
            o->default_value = val;
        } else {
            o->default_value = opt.default_val;
        }
    }
    std::function<std::expected<void, std::string>(const Runtime_value &)> val_fn;

    if (opt.validators_.empty())
    {
        val_fn = [](const auto &) -> std::expected<void, std::string> { return {}; };
    }
    else
    {
        // Capture the type-erased validator entries by value
        val_fn = [entries = opt.validators_, kind = storage_kind](const Runtime_value &rv) -> std::expected<void, std::string>
        {
            auto check_one = [&](const T &val) -> std::expected<void, std::string>
            {
                for (const auto &e : entries)
                    if (auto r = e.func(val); !r)
                        return r;
                return {};
            };

            // CASE 1: Container (Vector/Array)
            if (kind == Storage_kind::Vector || kind == Storage_kind::Array)
            {
                const auto &raw_vec = std::get<std::vector<CanonElem>>(rv);

                if constexpr (is_std_vector_v<T> || is_std_array_v<T>)
                {
                    T container;
                    using ValType = typename T::value_type;
                    if constexpr (is_std_vector_v<T>)
                    {
                        container.reserve(raw_vec.size());
                        for (const auto &e : raw_vec) container.push_back(static_cast<ValType>(e));
                    }
                    else
                    {
                        // std::array reconstruction
                        for (size_t i = 0; i < raw_vec.size() && i < std::tuple_size_v<T>; ++i)
                            container[i] = static_cast<ValType>(raw_vec[i]);
                    }
                    return check_one(container);
                }
                else
                {
                    // Scalar Multi
                    for (size_t i = 0; i < raw_vec.size(); ++i)
                    {
                        // static_cast on arrays is invalid, we reconstruct scalar here
                        T user_val = static_cast<T>(raw_vec[i]);
                        if (auto r = check_one(user_val); !r)
                            return std::unexpected(std::format("Item {}: {}", i, r.error()));
                    }
                    return {};
                }
            }
            // CASE 2: Scalar
            else
            {
                if constexpr (!is_std_array_v<T>)
                {
                    const auto &canon_val = std::get<CanonElem>(rv);
                    return check_one(static_cast<T>(canon_val));
                }
                return {};
            }
        };
    }

    // 4. Store Option
    std::vector<std::string> helps;
    for (auto &v : opt.validators_) helps.push_back(v.help);
    o->validate = val_fn;

    Runtime rt{};

    if (o->flags & (*Flags::O_Default))
        o->default_hints = arena_.str(std::format("{}", opt.default_val));

    if (storage_kind == Storage_kind::Vector)
        rt.runtime_value = std::vector<CanonElem>{};
    else if (storage_kind == Storage_kind::Array)
        rt.runtime_value = std::vector<CanonElem>(o->arity, CanonElem{});
    else
        rt.runtime_value = CanonElem{};

    options_.push_back(o);
    runtime_.push_back(rt);
    return id;
}

// -----------------------------------------------------------------------------
// IMPLEMENTATION: PARSER::PARSE
// -----------------------------------------------------------------------------

auto cl::Parser::parse(int argc, char *argv[]) -> std::expected<Parse_res, Parse_err>
{
    Arg_stream args(argc, argv);
    Parse_err err{};
    Parse_res res{};
    Parse_ctx ctx(args, err, res, this->cfg_);
    res.runtime_.resize(this->runtime_.size());
    std::copy(this->runtime_.begin(), this->runtime_.end(), res.runtime_.begin());

    while (!args.empty())
    {
        std::string_view tok = args.pop();;
        if (tok.starts_with("--"))
            this->handle_long_token(ctx, tok.substr(2));
        else if (tok.starts_with("-"))
            this->handle_short_token(ctx, tok.substr(1));
    }

    if (!err.errors.empty()) return std::unexpected(err);

    res.opt_info_.resize(this->options_.size());

    for (auto i{0uz}; i < this->options_.size(); i++)
    {
        Option * opt = this->options_[i];
        res.opt_info_[i].type    = opt->type;
        res.opt_info_[i].arity   = opt->arity;
        res.opt_info_[i].storage = opt->storage;

        if (res.runtime_[i].parsed) continue;

        if ((opt->flags & (*Flags::O_Default)))
            res.runtime_[i].runtime_value = opt->default_value;
    }

    return res;
}

template <typename Dest>
requires cl::Supported_Scalar_C<Dest>
inline auto cl::Parser::string_to_value(std::string_view s) -> std::expected<Dest, std::string>
{
    if constexpr (std::is_same_v<Dest, std::string> || std::is_same_v<Dest, std::string_view>)
    {
        return std::string(s);
    }
    else if constexpr (std::is_same_v<Dest, bool>)
    {
        for (auto &t : truthy_strs)
            if (s == t)
                return true;
        for (auto &f : falsy_strs)
            if (s == f)
                return false;

        return std::unexpected(std::format("Invalid boolean value: '{}'", s));
    }
    else if constexpr (std::is_integral_v<Dest>)
    {
        Dest v{};
        auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
        if (ec != std::errc{})
            return std::unexpected(std::format("Invalid integer: '{}'", s));
        return v;
    }
    else if constexpr (std::is_floating_point_v<Dest>)
    {
        Dest v{};
        auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
        if (ec != std::errc{})
            return std::unexpected(std::format("Invalid float: '{}'", s));
        return v;
    }
}

void cl::Parser::handle_long_token(Parse_ctx &ctx, std::string_view body)
{
    std::string_view key = body;
    std::optional<std::string_view> explicit_val = std::nullopt;
    ctx.curr_key = body;

    if (size_t eq_pos = body.find('='); eq_pos != std::string_view::npos)
    {
        if (this->cfg_.value_binding == Binding_type::Next)
            return ctx.error("Equals '=' binding is disabled by configuration.");
        key = body.substr(0, eq_pos);
        ctx.curr_key = key;
        explicit_val = body.substr(eq_pos + 1);
    }

    auto it = long_arg_to_id_.find(key);
    if (it == long_arg_to_id_.end())
        return ctx.error("Unknown option");

    acquire_value(ctx, options_[it->second], explicit_val);
}

void cl::Parser::handle_short_token(Parse_ctx &ctx, std::string_view body)
{
    if (body.empty())
        return;
    std::string_view key = body.substr(0, 1);
    ctx.curr_key = key;

    auto it = short_arg_to_id_.find(key);
    if (it == short_arg_to_id_.end())
        return ctx.error("Unknown flag -{}", key);

    Option *opt = options_[it->second];

    if (opt->arity == 0)
    {
        this->assign_true(ctx.res.runtime_[opt->id]);
        // If characters remain, handle them recursively
        if (body.size() > 1)
        {
            if (!ctx.cfg.allow_combined_short_flags)
                return ctx.error("Combined short flags are disabled.");

            this->add_short_combined(ctx, body.substr(1));
        }
        return;
    }

    std::optional<std::string_view> attached_val = std::nullopt;

    if (body.size() > 1)
    {
        // Check if user is trying to attach a value
        if (!ctx.cfg.allow_short_value_concat)
            return ctx.error("Concatenated values disabled.");

        // Handle edge case: -j=4 vs -j4
        // Only strip '=' if the binding config allows it (Equal or Both)
        if (body[1] == '=' && ctx.cfg.value_binding != Binding_type::Next)
            attached_val = body.substr(2);  // Skip flag and '='
        else
            attached_val = body.substr(1);  // Skip just the flag, rest is value
    }

    // Delegate to value acquisition (handles parsing/storage)
    acquire_value(ctx, opt, attached_val);
}

bool cl::Parser::add_short_combined(Parse_ctx &ctx, std::string_view body)
{
    for (auto x : body)
    {
        std::string_view curr_key{&x, 1};
        auto it = short_arg_to_id_.find(curr_key);
        ctx.curr_key = curr_key;
        if (it == short_arg_to_id_.end())
        {
            ctx.error("Unknown flag");
            return false;
        }
        Option *curr_opt = this->options_[it->second];

        if (curr_opt->arity > 0)
        {
            ctx.error("airty greater than zero, therefore can't be concatenated");
            return false;
        }

        this->assign_true(ctx.res.runtime_[curr_opt->id]);
    }
    return true;
}

bool cl::Parser::acquire_value(Parse_ctx &ctx, Option *opt, std::optional<std::string_view> explicit_val)
{
    auto &rt = ctx.res.runtime_[opt->id];

    if (rt.parsed && !(opt->flags & *Flags::O_Multi) && (opt->arity == 1))
    {
        switch (this->cfg_.repeated_scalar)
        {
            case cl::Repeated_scalar_policy::REJECT:
            {
                return false;
            }
            break;
            case Repeated_scalar_policy::FIRST:
            {
                return true;
            }
            break;
            case cl::Repeated_scalar_policy::LAST:
                break;
        }
    }

    if (opt->arity == 0)
    {
        if (explicit_val)
        {
            ctx.error("Flag '{}' cannot accept a value.", opt->names[0]);
            return false;
        }
        this->assign_true(rt);
        return true;
    }

    std::vector<std::string_view> raw_inputs{};
    if (explicit_val)
    {
        if ((opt->arity > 1) && (opt->list_cfg.type == List_type::Consecutive))
        {
            ctx.error("Consecutive arrays cannot use attached values.");
            return false;
        }
        // List type can only be Delimited now.
        // Split the input
        if (opt->arity > 1 || ((opt->flags & (*Flags::O_Multi)) && (opt->multi_cfg.type == Multi_type::Delimited)))
        {
            std::string_view delim = (opt->arity > 1) ? opt->list_cfg.delimiter : opt->multi_cfg.delimiter;

            for (auto &&part : std::views::split(*explicit_val, delim)) raw_inputs.emplace_back(std::string_view(part));
        }
        else
        {
            // Not array, not multi delimited
            // Only options: Scalar/Multi repeat
            raw_inputs.push_back(*explicit_val);
        }
    }
    else
    {
        // Fetch values from a stream.
        if (this->cfg_.value_binding != Binding_type::Next && this->cfg_.value_binding != Binding_type::Both)
        {
            ctx.error("No value for option, use '<key>='<value>'' to bind value");
            return false;
        }

        if (opt->arity > 1 && opt->list_cfg.type == List_type::Consecutive)
        {
            // Fetch N items
            for (size_t i{}; i < opt->arity; ++i)
            {
                if (ctx.args.empty())
                {
                    ctx.error("Not enough arguments for array.");
                    return false;
                }
                raw_inputs.push_back(ctx.args.pop());
            }
        }
        else
        {
            // Fetch 1 item (might be delimited string)
            if (ctx.args.empty())
            {
                ctx.error("Value not provided.");
                return false;
            }
            std::string_view val = ctx.args.pop();

            // Check for delimiter splitting again
            bool split_needed = (opt->arity > 1 && opt->list_cfg.type == List_type::Delimited) ||
                                ((opt->flags & *Flags::O_Multi) && opt->multi_cfg.type == Multi_type::Delimited);

            if (split_needed)
            {
                std::string_view delim = (opt->arity > 1) ? opt->list_cfg.delimiter : opt->multi_cfg.delimiter;
                for (auto &&part : std::views::split(val, delim)) raw_inputs.emplace_back(std::string_view(part));
            }
            else
            {
                // Either scalar or multi repeat
                raw_inputs.push_back(val);
            }
        }
    }
    this->inject_value(ctx, opt, raw_inputs);
    return true;
}

void cl::Parser::inject_value(Parse_ctx &ctx, Option *opt, std::span<const std::string_view> raw_values)
{
    auto &rt = ctx.res.runtime_[opt->id];

    // Ensure error reporting knows context (optional, but safe)
    ctx.curr_key = opt->names[0];
    bool result = false;

    std::visit(
        [&](auto &&storage)
        {
            using StorageT = std::decay_t<decltype(storage)>;

            // 1. Skip Monostate
            if constexpr (std::is_same_v<StorageT, std::monostate>)
                return;
            else
            {
                // 2. Safe Type Extraction
                using ElemT = typename cl::detail::get_inner_type<StorageT>::type;

                // 3. Process Values
                for (size_t i = 0; i < raw_values.size(); ++i)
                {
                    // Convert string to value
                    auto val_res = this->string_to_value<ElemT>(raw_values[i]);

                    if (!val_res)
                    {
                        ctx.error("{}", val_res.error());
                        return;
                    }

                    // 4. Storage Logic (Guarded by if constexpr)
                    if constexpr (is_std_vector_v<StorageT>)
                    {
                        // VECTOR / ARRAY LOGIC
                        bool is_fixed_array = (opt->storage == Storage_kind::Array);

                        if (is_fixed_array)
                        {
                            // Validate Array Size safely here
                            if (i >= storage.size())
                            {
                                if (!ctx.cfg.allow_empty_arrays)
                                    ctx.error("Internal Error: Target array size {} too small for input index {}.", storage.size(), i);
                                return;
                            }
                            storage[i] = *val_res;
                        }
                        else
                        {
                            storage.push_back(*val_res);
                            result = true;
                        }
                    }
                    else
                    {
                        // SCALAR LOGIC (int, bool, etc.)
                        storage = *val_res;
                        result = true;
                    }
                }
            }
        },
        rt.runtime_value);

    if (result)
    {
        if (auto valid = opt->validate(rt.runtime_value); !valid)
        {
            ctx.error("Validation failed: {}", valid.error());
            return;
        }
    }

    rt.parsed = true;
    rt.count++;
}

void cl::Parser::assign_true(Runtime& rt)
{
    rt.runtime_value = true;
    rt.count++;
    rt.parsed = true;
}

auto cl::Parser::print_help(std::ostream &os) -> void
{
    os << "help\n";
    //cl::debug::l1("Help: Generating Usage Info");
    //os << "Usage: " << name_ << " [options] [args]\n";
    //if (!description_.empty()) os << description_ << "\n";
    //os << "Options:\n";
    //
    //size_t max_width = 0;
    //for (const auto *opt : options_)
    //{
    //    if (opt->flags & *Flags::O_Hidden) continue;
    //
    //    size_t w = 0;
    //    if (!opt->names[1].empty())                           w += 2 + opt->names[1].size();
    //    if (!opt->names[1].empty() && !opt->names[0].empty()) w += 2;
    //    if (!opt->names[0].empty())                           w += 2 + opt->names[0].size();
    //    if (!opt->meta.empty())                               w += 1 + opt->meta.size();
    //    if (w > max_width)                                    max_width = w;
    //}
    //max_width += 4;
    //
    //for (const auto *opt : options_)
    //{
    //    if (opt->flags & *Flags::O_Hidden) continue;
    //
    //    std::string flags_part;
    //    if (!opt->names[1].empty())                           flags_part += std::format("-{}", opt->names[1]);
    //    if (!opt->names[1].empty() && !opt->names[0].empty()) flags_part += ", ";
    //    if (!opt->names[0].empty())                           flags_part += std::format("--{}", opt->names[0]);
    //    if (!opt->meta.empty())                               flags_part += std::format(" {}", opt->meta);
    //
    //    os << "  " << std::left << std::setw(max_width) << flags_part;
    //    os << opt->desc;
    //
    //    if (opt->flags & F_REQUIRED)
    //        os << " [Required]";
    //    if (!opt->default_val_str.empty() && opt->arity > 0)
    //        os << " [Default: " << opt->default_val_str << "]";
    //
    //    if (!opt->validator_helps.empty())
    //    {
    //        os << " {";
    //        bool f = true;
    //        for (const auto &h : opt->validator_helps)
    //        {
    //            if (!f) os << ", ";
    //            if (!h.empty()) os << h;
    //            else os << "Check";
    //            f = false;
    //        }
    //        os << "}";
    //    }
    //
    //    os << "\n";
    //}
    //os << "\n";
}

#endif // !CL_IMPLEMENTATION

#endif // !__CL_HPP_
