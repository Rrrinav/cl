#ifndef __CL_HPP_
#define __CL_HPP_

#warning "Not developed thing at all"

#include <sys/types.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <charconv>
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
#include <ranges>
#include <new>
#include <optional>
#include <print>
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
#include <variant>

// -----------------------------------------------------------------------------
// CONFIGURATION & MACROS
// -----------------------------------------------------------------------------

#ifndef CL_DEBUG_LEVEL
    #define CL_DEBUG_LEVEL 0
#endif

namespace cl::debug {

template <typename... Args>
inline auto l1(std::format_string<Args...> fmt, Args &&...args) -> void
{
    if constexpr (CL_DEBUG_LEVEL >= 1)
        std::println(stderr, "[CL_DBG: L1]: {}", std::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
inline auto l2(std::format_string<Args...> fmt, Args &&...args) -> void
{
    if constexpr (CL_DEBUG_LEVEL >= 2)
        std::println(stderr, "[CL_DBG: L2]:   -> {}", std::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
inline auto l3(std::format_string<Args...> fmt, Args &&...args) -> void
{
    if constexpr (CL_DEBUG_LEVEL >= 3)
        std::println(stderr, "[CL_DBG: L3]:         -> {}", std::format(fmt, std::forward<Args>(args)...));
}

inline auto todo(std::string &&s, std::source_location loc = std::source_location::current())
{
    std::println("{}:{}:{}: {}", loc.file_name(), loc.line(), loc.column(), s);
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

    explicit Arena(std::size_t block_size = 64 * 1024) : block_size_(block_size) { 
        cl::debug::l3("Arena: Initializing with block size {}", block_size);
        add_block();
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    ~Arena()
    {
        cl::debug::l2("Arena: Destructing. Objects: {}, Blocks: {}", destructors_.size(), blocks_.size());
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
            cl::debug::l3("Arena: Block full. Allocating new block for request of {} bytes", n);
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

constexpr auto name(std::string_view s, std::string_view l, std::source_location loc = std::source_location::current())
{
    return [s, l, loc]<typename Opt>(Opt& o) constexpr { o.args[0] = s; o.args[1] = l; o.loc = loc; };
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
        o.multi_cfg = {t, d};
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
        std::string message;
        std::string_view option_name;
    };
    std::vector<err> errors{};

    template <typename... Args>
    auto push_err(const std::string& option, std::format_string<Args...> fmt, Args &&...args)
    {
        return this->errors.push_back({option, std::format(fmt, std::forward<Args>(args)...)});
    }

    template <typename... Args>
    auto push_err(std::string_view option, std::format_string<Args...> fmt, Args &&...args)
    {
        return this->push_err(std::string(option), fmt, std::forward<Args>(args)...);
    }
};

using Runtime_value = std::variant<
std::string, long long, bool, double,
std::vector<std::string>,
std::vector<long long>,
std::vector<bool>,
std::vector<double>,
std::monostate
>;

struct Runtime
{
    Runtime_value runtime_value{std::monostate()};
    bool seen{false};
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
    std::vector<std::string> falsy_strs  = { "n", "false", "no", "f" };

private:
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

    template<typename T, typename ...Configs>
    requires ((Configurer<Configs, T> && ...) && Parsable_Type_C<T>)
    auto add(Configs&&... confs) -> Opt_id
    {
        Opt<T> opt;
        (confs(opt), ...);
        return this->add_impl(opt);
    }

    auto parse(int argc, char *argv[]) -> std::expected<Parse_res, std::string>;
    auto print_help(std::ostream& os = std::cout) -> void;

    void synchronize(Arg_stream& args)
    {
        while (!args.empty())
        {
            auto t = args.peek();
            // Stop if we see something that looks like a flag
            if (t)
            {
                if (t->starts_with("--")) return;
                if (t->starts_with("-") && t->size() == 2 && std::isalpha((*t)[1])) return;
            }
            args.pop();
        }
    }

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
    static inline auto string_to_value(std::string_view s) -> std::expected<Dest, std::string>
    {
        if constexpr (std::is_same_v<Dest, std::string> || std::is_same_v<Dest, std::string_view>)
        {
            return std::string(s);
        }
        else if constexpr (std::is_same_v<Dest, bool>)
        {
            if (s == "true" || s == "1" || s == "on" || s == "yes" || s == "y") return true;
            if (s == "false" || s == "0" || s == "off" || s == "no" || s == "n") return false;
            cl::debug::l3("ParseValue: Failed bool conversion for '{}'", s);
            return std::unexpected(std::format("Invalid boolean value: '{}'", s));
        }
        else if constexpr (std::is_integral_v<Dest>)
        {
            Dest v{};
            auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
            if (ec != std::errc{}) {
                cl::debug::l3("ParseValue: Failed int conversion for '{}'", s);
                return std::unexpected(std::format("Invalid integer: '{}'", s));
            }
            return v;
        }
        else if constexpr (std::is_floating_point_v<Dest>)
        {
            Dest v{};
            auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
            if (ec != std::errc{})
            {
                cl::debug::l3("ParseValue: Failed float conversion for '{}'", s);
                return std::unexpected(std::format("Invalid float: '{}'", s));
            }
            return v;
        }
    }

    auto assign_value_scalar(Runtime& _runtime, Parse_err & _err, Opt_type type, std::string_view s) -> std::expected<void, std::string>
    {
        _runtime.seen = true;
        _runtime.count++;
        switch (type)
        {
            case Opt_type::Bool:
            {
                if (auto typed_value = string_to_value<bool>(s); typed_value)
                    _runtime.runtime_value = *typed_value;
                else
                    return std::unexpected(typed_value.error());

                return {};
            } break;
            case Opt_type::Int:
            {
                if (auto typed_value = string_to_value<long long>(s); typed_value)
                    _runtime.runtime_value = *typed_value;
                else
                    return std::unexpected(typed_value.error());

                return {};
            } break;
            case Opt_type::Float:
            {
                if (auto typed_value = string_to_value<double>(s); typed_value)
                    _runtime.runtime_value = *typed_value;
                else
                    return std::unexpected(typed_value.error());

                return {};
            } break;
            case Opt_type::Str:
            {
                if (auto typed_value = string_to_value<std::string>(s); typed_value)
                    _runtime.runtime_value = *typed_value;
                else
                    return std::unexpected(typed_value.error());

                return {};
            } break;
            default: this->panic("Unreachable: Unknown type was detected.");
        }
        std::unreachable();
    }

    inline auto push_scalar(Runtime& _rt, Parse_err& _err, Option *opt, std::string_view sv) -> std::expected<void, std::string>
    {
        switch (opt->type) {
            case Opt_type::Int: {
                auto& vec = std::get<cl::List<cl::Num>>(_rt.runtime_value);
                if (auto res = this->string_to_value<cl::Num>(sv); !res)
                    return std::unexpected(res.error());
                else
                    vec.push_back(res.value());
            }break;

            case Opt_type::Float: {
                auto& vec = std::get<cl::List<cl::Fp_Num>>(_rt.runtime_value);
                if (auto res = this->string_to_value<cl::Fp_Num>(sv); !res)
                    return std::unexpected(res.error());
                else
                    vec.push_back(*res);
            }break; 

            case Opt_type::Str: {
                std::get<cl::List<cl::Text>>(_rt.runtime_value).push_back(std::string(sv));
            }break; 

            default: this->panic("Unreachable");
        }
        return {};
    }

    inline auto assign_vec(Runtime& _runtime, Parse_err& _err, Opt_type type, std::vector<std::string> s) -> std::expected<void, std::string>
    {
        _runtime.seen = true;
        _runtime.count++;
        switch (type)
        {
            case Opt_type::Int:
            {
                auto vec = std::get<std::vector<cl::Num>>(_runtime.runtime_value);
                cl::asrt::t((vec.size() >= s.size()), "Vec size mismatch");
                for ( int i = 0; i < s.size(); i++)
                {
                    if (auto typed_value = string_to_value<cl::Num>(s[i]); typed_value)
                        vec[i] = *typed_value;
                    else
                        return std::unexpected(typed_value.error());
                }
                return {};
            } break;
            case Opt_type::Float:
            {
                auto vec = std::get<std::vector<cl::Fp_Num>>(_runtime.runtime_value);
                cl::asrt::t((vec.size() == s.size()), "Vec size mismatch");
                for ( int i = 0; i < s.size(); i++)
                {
                    if (auto typed_value = string_to_value<cl::Fp_Num>(s[i]); typed_value)
                        vec[i] = *typed_value;
                    else
                        return std::unexpected(typed_value.error());
                }
                return {};
            } break;
            case Opt_type::Str:
            {
                auto vec = std::get<std::vector<cl::Text>>(_runtime.runtime_value);
                cl::asrt::t((vec.size() == s.size()), "Vec size mismatch");
                for ( int i = 0; i < s.size(); i++)
                {
                    if (auto typed_value = string_to_value<cl::Text>(s[i]); typed_value)
                        vec[i] = *typed_value;
                    else
                        return std::unexpected(typed_value.error());
                }
                return {};
            } break;
            default: this->panic("Unreachable: Unknown type was detected.");
        }
        std::unreachable();
    }

    auto fill_delimited_arr(const std::string& delimiter, std::string_view s, const Option * opt, Runtime& rt, Parse_err& err) -> std::expected<void, std::string>
    {
        auto splitted = std::views::split(s, delimiter);

        switch (opt->type)
        {
            case Opt_type::Int: {
                std::size_t count = 0;
                auto &res_vec = std::get<std::vector<cl::Num>>(rt.runtime_value);
                auto &def_vec = std::get<std::vector<cl::Num>>(opt->default_value);
                cl::asrt::t((res_vec.size() == opt->arity && res_vec.size() == def_vec.size()), "Vector size not same as arity, some library error.");
                if (this->cfg_.allow_empty_arrays)
                {
                    if (opt->flags & (*Flags::O_Default))
                        std::copy(def_vec.begin(), def_vec.end(), res_vec.begin());
                    else
                        std::fill(res_vec.begin(), res_vec.end(), cl::Num{});
                }

                for (auto &&part : splitted)
                {
                    std::string s = std::string(part.begin(), std::ranges::distance(part));
                    if (s.empty())
                    {
                        if (this->cfg_.allow_empty_arrays)
                            continue;
                        else
                            return std::unexpected("Empty elements in arrays now allowed");
                    }
                    if (auto ans = this->string_to_value<cl::Num>(s); ans)
                        res_vec[count] = *ans;
                    else
                        return std::unexpected(ans.error());
                    ++count;
                }
                if ((count < opt->arity - 1) && !this->cfg_.allow_empty_arrays) return std::unexpected("Empty elements in arrays now allowed");
            } break;

            case Opt_type::Float: {
                std::size_t count = 0;
                auto &res_vec = std::get<std::vector<cl::Fp_Num>>(rt.runtime_value);
                auto &def_vec = std::get<std::vector<cl::Fp_Num>>(opt->default_value);
                cl::asrt::t((res_vec.size() == opt->arity && res_vec.size() == def_vec.size()), "Vector size not same as arity, some library error.");

                if (this->cfg_.allow_empty_arrays)
                {
                    if (opt->flags & (*Flags::O_Default))
                        std::copy(def_vec.begin(), def_vec.end(), res_vec.begin());
                    else
                        std::fill(res_vec.begin(), res_vec.end(), cl::Fp_Num{});
                }

                for (auto &&part : splitted)
                {
                    std::string s = std::string(part.begin(), std::ranges::distance(part));
                    if (s.empty())
                    {
                        if (this->cfg_.allow_empty_arrays)
                            continue;
                        else
                            return std::unexpected("Empty elements in arrays now allowed");
                    }
                    if (auto ans = this->string_to_value<cl::Fp_Num>(s); ans)
                        res_vec[count] = *ans;
                    else
                        return std::unexpected(ans.error());
                    ++count;
                }
                if ((count < opt->arity - 1) && !this->cfg_.allow_empty_arrays) return std::unexpected("Empty elements in arrays now allowed");
            } break;

            case Opt_type::Str: {
                std::size_t count = 0;
                auto &res_vec = std::get<std::vector<cl::Text>>(rt.runtime_value);
                auto &def_vec = std::get<std::vector<cl::Text>>(opt->default_value);
                cl::asrt::t((res_vec.size() == opt->arity && res_vec.size() == def_vec.size()), "Vector size not same as arity, some library error.");

                if (this->cfg_.allow_empty_arrays)
                {
                    if (opt->flags & (*Flags::O_Default))
                        std::copy(def_vec.begin(), def_vec.end(), res_vec.begin());
                    else
                        std::fill(res_vec.begin(), res_vec.end(), cl::Text{});
                }

                for (auto &&part : splitted)
                {
                    std::string s = std::string(part.begin(), std::ranges::distance(part));
                    res_vec[count] = s;
                    ++count;
                }
                if ((count < opt->arity - 1) && !this->cfg_.allow_empty_arrays) return std::unexpected("Empty elements in arrays now allowed");
            } break;
            default: this->panic("Unreachable: unknown type of vector");
        }
        return {};
    }

    auto fill_arr_next(const Option * opt, Runtime& rt, Arg_stream & args, Parse_err& err) -> std::expected<void, std::string>
    {

        switch (opt->type) {
            case Opt_type::Int: {
                auto &res_vec = std::get<std::vector<cl::Num>>(rt.runtime_value);
                auto &def_vec = std::get<std::vector<cl::Num>>(opt->default_value);
                cl::asrt::t((res_vec.size() == opt->arity && res_vec.size() == def_vec.size()), "Vector size not same as arity, some library error.");
                if (this->cfg_.allow_empty_arrays)
                {
                    if (opt->flags & (*Flags::O_Default))
                        std::copy(def_vec.begin(), def_vec.end(), res_vec.begin());
                    else
                        std::fill(res_vec.begin(), res_vec.end(), cl::Num{});
                }

                std::size_t amount = opt->arity;
                std::size_t ind  = 0;

                std::string value;
                while(!args.empty() && ind < amount)
                {
                    value = args.pop();
                    if (value.empty())
                    {
                        if (this->cfg_.allow_empty_arrays)
                            continue;
                        else
                            return std::unexpected("Empty elements in arrays now allowed");
                    }
                    if (auto ans = this->string_to_value<cl::Num>(value); ans)
                        res_vec[ind] = *ans;
                    else
                        return std::unexpected(ans.error());
                    ind++;
                }
                if ((ind < opt->arity - 1) && !this->cfg_.allow_empty_arrays) return std::unexpected("Empty elements in arrays now allowed");
            } break;
            case Opt_type::Float: {
                auto &res_vec = std::get<std::vector<cl::Fp_Num>>(rt.runtime_value);
                auto &def_vec = std::get<std::vector<cl::Fp_Num>>(opt->default_value);

                cl::asrt::t((res_vec.size() == opt->arity && res_vec.size() == def_vec.size()), "Vector size not same as arity, some library error.");

                if (this->cfg_.allow_empty_arrays)
                {
                    if (opt->flags & (*Flags::O_Default))
                        std::copy(def_vec.begin(), def_vec.end(), res_vec.begin());
                    else
                        std::fill(res_vec.begin(), res_vec.end(), cl::Num{});
                }

                std::size_t amount = opt->arity;
                std::size_t ind  = 0;

                std::string value;
                while(!args.empty() && ind < amount)
                {
                    value = args.pop();
                    if (value.empty())
                    {
                        if (this->cfg_.allow_empty_arrays)
                            continue;
                        else
                            return std::unexpected("Empty elements in arrays now allowed");
                    }
                    if (auto ans = this->string_to_value<cl::Fp_Num>(value); ans)
                        res_vec[ind] = *ans;
                    else
                        return std::unexpected(ans.error());
                    ind++;
                }
                if ((ind < opt->arity - 1) && !this->cfg_.allow_empty_arrays) return std::unexpected("Empty elements in arrays now allowed");
            } break;

            case Opt_type::Str: {
                auto &res_vec = std::get<std::vector<cl::Text>>(rt.runtime_value);
                auto &def_vec = std::get<std::vector<cl::Text>>(opt->default_value);

                cl::asrt::t((res_vec.size() == opt->arity && res_vec.size() == def_vec.size()), "Vector size not same as arity, some library error.");

                if (this->cfg_.allow_empty_arrays)
                {
                    if (opt->flags & (*Flags::O_Default))
                        std::copy(def_vec.begin(), def_vec.end(), res_vec.begin());
                    else
                        std::fill(res_vec.begin(), res_vec.end(), cl::Num{});
                }

                std::size_t amount = opt->arity;
                std::size_t ind  = 0;

                std::string value;
                while(!args.empty() && ind < amount)
                {
                    value = args.pop();
                    res_vec[ind] = value;
                    ind++;
                }
                if ((ind < opt->arity - 1) && !this->cfg_.allow_empty_arrays) return std::unexpected("Empty elements in arrays now allowed");
            } break;
            default: this->panic("Unreachable");
        }
        return {};
    }


    template<typename CanonElem, typename ValidatorVec>
    static auto validate_storage(void* storage, Storage_kind kind, std::size_t arity, const ValidatorVec& validators) -> std::expected<void, std::string>
    {
        cl::debug::l3("ValidateStorage: Checking constraints for storage kind {}", static_cast<int>(kind));
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

    auto handle_multi()
    {
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

            if (!e.option_name.empty())
                out = std::format_to(out, "\n  option: {}", e.option_name);

            if (i + 1 < pe.errors.size())
                out = std::format_to(out, "\n");
        }

        return out;
    }
};
namespace cl {

// -----------------------------------------------------------------------------
// IMPLEMENTATION: PARSER::ADD
// -----------------------------------------------------------------------------

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

template <typename T>
requires cl::Parsable_Type_C<T>
inline auto cl::Parser::add_impl(Opt<T> opt) -> Opt_id
{
    cl::debug::l1("Add: Registering Option [long='{}', short='{}']", opt.args[0], opt.args[1]);

    if (opt.args[0].empty() && opt.args[1].empty()) panic("Option must have at least one name (short or long)");

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
        cl::debug::l2("Mapped long name '--{}' to ID {}", name, id);
    }

    if (!opt.args[this->sn_index].empty())
    {
        auto name = arena_.str(opt.args[this->sn_index]);
        cl::asrt::t((name.size() == 1), "Short option '{}' must be 1 char", name);
        cl::asrt::t((std::isalpha(name[0])), "Short option '{}' must be a letter or alphabet.", name);
        if (short_arg_to_id_.contains(name)) panic("Duplicate option: -{}", name);
        short_arg_to_id_.emplace(name, id);
        cl::debug::l2("Mapped short name '-{}' to ID {}", name, id);
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

    cl::debug::l2("Type Analysis: UserT={}, CanonicalT={}, OptType={}", type_name<T>(), type_name<CanonElem>(), target_enum);

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
    cl::debug::l2("Config: Arity={}, Storage={}, Flags=0b{:b}", arity, storage_kind, opt.flags);

    Runtime_value val;

    if (storage_kind == Storage_kind::Vector)
    {
        cl::debug::l3("Storage: Allocating std::vector<{}>", typeid(CanonElem).name());
        val = std::vector<CanonElem>{};
    }
    else if constexpr (is_std_array_v<T>)
    {
        constexpr size_t N = std::tuple_size_v<T>;
        cl::debug::l3("Storage: Allocating std::array<{}, {}>", typeid(CanonElem).name(), N);
        std::vector<CanonElem> temp_vec(N, {});
        val = std::move(temp_vec);
    }
    else
    {
        cl::debug::l3("Storage: Allocating Scalar {}", typeid(CanonElem).name());
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

// TODO: Make this thing totally totally cleaner.
auto cl::Parser::parse(int argc, char *argv[]) -> std::expected<Parse_res, std::string>
{
    cl::debug::l1("Parse {}: Start (argc={})",this->name_, argc);
    Arg_stream args(argc, argv);
    bool stop_parsing = false;
    Parse_err err{};
    Parse_res res{};

    bool is_next_binding_allowed = (this->cfg_.value_binding == Binding_type::Next) || (this->cfg_.value_binding == Binding_type::Both);

    res.runtime_.resize(this->runtime_.size());
    std::copy(this->runtime_.begin(), this->runtime_.end(), res.runtime_.begin());

    // WARN: Serious lifetime issues here.
    auto to_sv = [](auto &&r) { return std::string(&*r.begin(), std::ranges::distance(r)); };
    // 1. Tokenization Loop
    while (!args.empty())
    {
        auto opt_tok = args.pop();

        std::string_view tok = opt_tok;
        cl::debug::l2("ParseLoop: Processing Token '{}'", tok);

        if (stop_parsing)
        {
            cl::debug::l3("Storing as positional (stop_parsing=true)");
            cl::debug::todo("Positionals after stop parsing.");
            //positional_args_.push_back(tok);
            continue;
        }

        if (tok == "--" && cfg_.stop_on_double_dash)
        {
            cl::debug::l2("Found '--', stopping option parsing");
            stop_parsing = true;
            continue;
        }

        // --- Long Option ---
        if (tok.starts_with("--"))
        {
            std::string_view body = tok.substr(2);
            std::string_view key = body;
            std::string_view value;
            bool has_eq = false;

            if (auto eq = body.find('='); eq != std::string_view::npos)
            {
                cl::debug::l2("Body has '='");
                if (this->cfg_.value_binding == Binding_type::Next)
                {
                    err.push_err(body, "Equals syntax not allowed");
                }
                cl::debug::l3("Value binding with equals alowed");
                key = body.substr(0, eq);
                value = body.substr(eq + 1);
                has_eq = true;
                cl::debug::l2("key: {}, value: {}", key, value);
            }

            if (!long_arg_to_id_.contains(key))
            {
                err.push_err(std::string(key), "{}", "Unknown option");
            }
            auto id = long_arg_to_id_[key];
            auto *opt = options_[id];
            auto &rt = res.runtime_[id];

            cl::debug::l2("Matched Long Option: --{} (ID: {}), (Option: {})", key, id, *opt);

            if (opt->arity == 0)
            {
                if (res.runtime_[opt->id].seen) continue;

                if (has_eq) err.push_err(body, "A flag can't have a value bounded to it using '='");
                cl::debug::l3("opt->arity = 0");
                // TODO: Write a function to do this bs.
                rt.runtime_value = true;
                rt.seen = true;
                rt.count++;
                //std::println("Set flag: --{}", body);
                cl::debug::l2("Set flag as true");
                continue;
            }

            if (opt->arity == 1)
            {
                cl::debug::l3("opt->arity = 1");

                if (!has_eq)
                {
                    cl::debug::l3("Doesnt have equals");
                    if (!is_next_binding_allowed) std::unexpected("Bind value using '='.");
                    if (args.empty()) return std::unexpected(std::format("No value provided for flag --{}", body));
                    value = args.pop();
                }
                cl::debug::l3("Final value: {}", value);

                if (opt->flags & *Flags::O_Multi)
                {
                    cl::debug::l2("Option type: Multi");
                    switch (opt->multi_cfg.type) {
                        case Multi_type::Repeat: {
                            cl::debug::l2("Multi type: Repeat");
                            if (auto value_injection_res = this->push_scalar(rt, err, opt, value); !value_injection_res)
                                return std::unexpected(value_injection_res.error());
                        } break;

                        case Multi_type::Delimited: {
                            cl::debug::l2("Multi type: Delimited");

                            auto splitted = std::views::split(value, opt->multi_cfg.delimiter);
                            cl::debug::l2("Splitted value with delimiter: {} : {}", opt->multi_cfg.delimiter, splitted);

                            for (auto && s : splitted)
                            {
                                // TODO: Use 's'
                                std::string str = std::string(s.begin(), std::ranges::distance(s));
                                if (auto value_injection_res = this->assign_value_scalar(rt, err, opt->type, str); !value_injection_res)
                                    return std::unexpected(value_injection_res.error());
                            }
                        } break;
                        default: this->panic("Unreacheble point.");
                    }
                    continue;
                }
                else
                {
                    if (!has_eq)
                    {
                        cl::debug::l3("Doesn't have equals");
                        if (!is_next_binding_allowed) std::unexpected("Bind value using '='.");
                        if (args.empty()) return std::unexpected(std::format("No value provided for flag --{}", body));
                        value = args.pop();
                    }
                    cl::debug::l2("Final value: {}", value);

                    switch (this->cfg_.repeated_scalar) {

                        case Repeated_scalar_policy::REJECT: {

                            if (rt.seen) return std::unexpected("Value already assigned.");
                            if (auto value_injection_res = this->assign_value_scalar(rt, err, opt->type, value); !value_injection_res)
                                return std::unexpected(value_injection_res.error());

                        } break;

                        case Repeated_scalar_policy::FIRST: {

                            if (rt.seen) continue;
                            if (auto value_injection_res = this->assign_value_scalar(rt, err, opt->type, value); !value_injection_res)
                                return std::unexpected(value_injection_res.error());

                        } break;

                        case Repeated_scalar_policy::LAST: {

                            if (auto value_injection_res = this->assign_value_scalar(rt, err, opt->type, value); !value_injection_res)
                                return std::unexpected(value_injection_res.error());

                        } break;

                        default: this->panic("Unreacheble point.");
                    }
                }
            }

            if (opt->arity > 1)
            {
                cl::debug::l2("Arity > 1: {}", opt->arity);
                if (has_eq)
                {
                    cl::debug::l2("Has equals");
                    if (opt->list_cfg.type == List_type::Consecutive) std::unexpected("");

                    if (auto res = this->fill_delimited_arr(opt->list_cfg.delimiter, value, opt, rt, err); !res) return std::unexpected(res.error());

                    rt.seen = true;
                    rt.count++;
                    continue;
                }
                else
                {
                    if (opt->list_cfg.type == List_type::Consecutive)
                    {
                        cl::debug::l2("List type: Consecutive.");
                        if (auto fill_res = this->fill_arr_next(opt, rt, args, err); !fill_res )
                            return std::unexpected(fill_res.error());

                        rt.count++;
                        rt.seen = true;
                        continue;
                    }
                    else
                    {
                        cl::debug::l2("List type: Delimited.");
                        value = args.pop();

                        if (auto res = this->fill_delimited_arr(opt->list_cfg.delimiter, value, opt, rt, err); !res) return std::unexpected(res.error());

                        rt.seen = true;
                        rt.count++;
                        continue;
                    }
                }
            }
            continue;
        }

        if (tok.starts_with("-") && tok.size() > 1)
        {
            cl::debug::l1("Small token: {} starting with '-'", tok);

            // We have a single flag: something like -v
            if (tok.size() == 2)
            {
                std::string_view flag_body = tok.substr(1);
                cl::debug::l2("Token size: 2: body: {}", flag_body);
                // Get option and runtime
                Option* opt{};

                if (auto it = this->short_arg_to_id_.find(flag_body); it == this->short_arg_to_id_.end())
                    return std::unexpected(std::format("The option: -{} doesn't exist.", flag_body));
                else
                    opt = this->options_[it->second];

                Runtime& rt = res.runtime_[opt->id];

                cl::debug::l3("Found corresponding option:  {}", *opt);

                if(opt->arity == 0)
                {
                    cl::debug::l2("Arity: 0: marking true.");
                    rt.runtime_value = true;
                }

                if(opt->arity == 1)
                {
                    cl::debug::l2("Arity = 1");
                    if (opt->flags & *Flags::O_Multi)
                    {
                        cl::debug::l2("Multi flag");
                        switch (opt->multi_cfg.type) {

                            case Multi_type::Repeat: {
                                cl::debug::l2("Multi type: Repeat.");
                                // WARN: This may cause a serious bug; the size thing because empty string might be given too.
                                if (auto val = args.pop(); val.size())
                                {
                                    if (auto injection_res = this->push_scalar(rt, err, opt, val); !injection_res)
                                        return std::unexpected(injection_res.error());
                                }
                                return std::unexpected(std::format("No value provided for option: {}", flag_body));
                            } break;

                            case Multi_type::Delimited: {
                                if (auto val = args.pop(); val.size())
                                {
                                    for (auto &&_v : std::views::split(val, opt->multi_cfg.delimiter))
                                    {
                                        auto v = std::string_view{_v.begin(), _v.end()};
                                        if (auto injection_res = this->push_scalar(rt, err, opt, v); !injection_res)
                                            return std::unexpected(injection_res.error());
                                    }
                                }
                            } break;
                            default: this->panic( "Code reached unreachable point.");
                        }
                        continue;
                    }
                    else
                    {
                        switch (this->cfg_.repeated_scalar) {
                            // TODO: Make if(!seen) condition outside to avoid too much branching inside
                            case Repeated_scalar_policy::REJECT: {
                                if (rt.seen) return std::unexpected(std::format("scalar option: {} specified twice, it is not allowed", flag_body));
                                if (auto val = args.pop(); val.size())
                                {
                                    if (auto injection_res = this->assign_value_scalar(rt, err, opt->type, val); !injection_res)
                                        return std::unexpected(injection_res.error());
                                }
                            } break;
                            case Repeated_scalar_policy::FIRST: {
                                if (rt.seen) continue;
                                if (auto val = args.pop(); val.size())
                                {
                                    if (auto injection_res = this->assign_value_scalar(rt, err, opt->type, val); !injection_res)
                                        return std::unexpected(injection_res.error());
                                }
                            }continue; break;
                            case Repeated_scalar_policy::LAST: {
                                if (auto val = args.pop(); val.size())
                                {
                                    if (auto injection_res = this->assign_value_scalar(rt, err, opt->type, val); !injection_res)
                                        return std::unexpected(injection_res.error());
                                }
                            } break;
                            default: this->panic("Code reached unreachable point.");
                        }
                        rt.seen = true;
                        rt.count++;
                        continue;
                    }
                }
                if (opt->arity > 1)
                {
                    if (opt->list_cfg.type == List_type::Consecutive)
                    {
                        if (auto injection_res = this->fill_arr_next(opt, rt, args, err); !injection_res)
                            std::unexpected(injection_res.error());
                    }
                    else
                    {
                        std::string_view value_str{};
                        if ((value_str.empty() && (!this->cfg_.allow_empty_arrays))) return std::unexpected("Not enough elements provided for array.");
                        value_str = args.pop();
                        if (auto injection_res = this->fill_delimited_arr(opt->list_cfg.delimiter, value_str, opt, rt, err); !injection_res)
                            return std::unexpected(injection_res.error());
                    }
                    continue;
                }
            }
            else
            {
                std::string_view first_flag = tok.substr(1, 1);
                Option * opt{};

                if (auto it = this->short_arg_to_id_.find(first_flag); it != this->short_arg_to_id_.end())
                    opt = this->options_[it->second];
                else
                    return std::unexpected(std::format("No such flag as: -{}", first_flag));

                Runtime& rt = res.runtime_[opt->id];

                if (opt->arity == 0)
                {
                    cl::debug::l2("Arity of first one: 0");
                    if (!this->cfg_.allow_combined_short_flags) return std::unexpected("Combining short flags is not allowed");
                    for (size_t i = 1; i < tok.size(); ++i)
                    {
                        std::string_view key{&tok[i], 1};

                        auto it = short_arg_to_id_.find(key);
                        if (it == short_arg_to_id_.end())
                            return std::unexpected(std::format("No such flag as: -{}", tok[i]));

                        Option *opt = options_[it->second];
                        if (opt->arity != 0)
                            return std::unexpected(std::format("Arity of flag: -{} is not 0 thus it can't be concatenated.", key));

                        rt.runtime_value = true;
                    }
                    rt.seen = true;
                    rt.count++;
                    continue;
                }
                else if (opt->arity == 1)
                {
                    if (!this->cfg_.allow_short_value_concat) return std::unexpected("Concatenated short values not allowed");

                    std::string_view value_str{};

                    // TODO: Decide on how to handle the case when user enters '-x='.
                    // Though this wont crash in that case too, it will only crash if I put value > 3
                    if (this->cfg_.value_binding == Binding_type::Equal && tok[2] == '=')
                        value_str = tok.substr(3);
                    else
                        value_str = tok.substr(2);

                    if (opt->flags & *Flags::O_Multi)
                    {
                        switch (opt->multi_cfg.type) {
                            case Multi_type::Repeat: {
                                if (auto injection_res = this->push_scalar(rt, err, opt, value_str); !injection_res)
                                    return std::unexpected(injection_res.error());
                            } break;
                            case Multi_type::Delimited: {
                                // TODO: Handle this case in the add function.
                                for (auto &&_v : std::views::split(value_str, opt->multi_cfg.delimiter))
                                {
                                    auto v = std::string_view{_v.begin(), _v.end()};
                                    if (auto injection_res = this->push_scalar(rt, err, opt, v); !injection_res)
                                        return std::unexpected(injection_res.error());
                                }
                            } break;
                            default: this->panic("Code reached unreachable point.");
                        }
                        rt.seen = true;
                        rt.count++;
                        continue;
                    }
                    else
                    {
                        switch (this->cfg_.repeated_scalar) {
                            case Repeated_scalar_policy::REJECT: {
                                if (rt.seen) return std::unexpected(std::format("scalar option: {} specified twice, it is not allowed", first_flag));
                                if (auto injection_res = this->assign_value_scalar(rt, err, opt->type, value_str); !injection_res)
                                    return std::unexpected(injection_res.error());
                            } break;
                            case Repeated_scalar_policy::FIRST: {
                                if (rt.seen) continue;
                                if (auto injection_res = this->assign_value_scalar(rt, err, opt->type, value_str); !injection_res)
                                    return std::unexpected(injection_res.error());
                            }break;
                            case Repeated_scalar_policy::LAST: {
                                if (auto injection_res = this->assign_value_scalar(rt, err, opt->type, value_str); !injection_res)
                                    return std::unexpected(injection_res.error());
                            } break;
                            default: this->panic("Unreachable");
                        }
                        rt.seen = true;
                        rt.count++;
                        continue;
                    }
                }
                // Only allowed way to enter a value here is: "-a1,2,3", "-a=1,2,3"
                std::string_view value_str{};
                // TODO: Decide on how to handle the case when user enters '-x='.
                // Though this wont crash in that case too, it will only crash if I put value > 3
                if (this->cfg_.value_binding == Binding_type::Equal && tok[2] == '=')
                    value_str = tok.substr(3);
                else
                    value_str = tok.substr(2);
                if (rt.seen) return std::unexpected(std::format("Flag: -{} can't appear twice.", first_flag));

                // TODO: Handle all configuration errors before parsing starts.
                auto splitted = std::views::split(value_str, opt->list_cfg.delimiter);

                auto to_sv = [](auto &&r) { return std::string_view(&*r.begin(), std::ranges::distance(r)); };

                if (auto injection_res = this->fill_delimited_arr(opt->list_cfg.delimiter, value_str, opt, rt, err); !injection_res)
                    return std::unexpected(injection_res.error());

                rt.seen = true;
                rt.count++;
                continue;
            }
        }

        cl::debug::l3("Storing as positional: '{}'", tok);
        //positional_args_.push_back(tok);
    }

    res.opt_info_.resize(this->options_.size());

    for (auto i{0uz}; i < this->options_.size(); i++)
    {
        res.opt_info_[i].type = this->options_[i]->type;
        res.opt_info_[i].arity = this->options_[i]->arity;
        res.opt_info_[i].storage = this->options_[i]->storage;
    }

    cl::debug::l1("Parse: Success");
    return res;
}

// -----------------------------------------------------------------------------
// IMPLEMENTATION: PRINT HELP
// -----------------------------------------------------------------------------

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

} // namespace cl

#endif // !__CL_HPP_
