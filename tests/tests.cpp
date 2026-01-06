#include <expected>
#define CL_IMPLEMENTATION
#include "cl.hpp"
#include <exception>
#include <iomanip>
#include <ostream>
#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include <cmath>

// -----------------------------------------------------------------------------
// HELPER: Fake Command Line Arguments
// -----------------------------------------------------------------------------
struct Args
{
    std::vector<std::string> args;
    std::vector<char *> argv;

    // Use initializer list to simulate argv
    Args(std::initializer_list<std::string> list) : args(list)
    {
        // First arg is usually program name
        for (auto &s : args) argv.push_back(const_cast<char *>(s.data()));
    }

    int argc() const { return static_cast<int>(argv.size()); }
    char **ptr() { return argv.data(); }
};

constexpr auto width = 55;
constexpr auto passed_str   = "Passed";
constexpr auto failed_str   = "Failed";

namespace ansi
{
inline constexpr const char *reset = "\033[0m";

// Absolute RGB colors (NOT theme-dependent)
inline constexpr const char *red = "\033[38;2;255;0;0m";
inline constexpr const char *green = "\033[38;2;0;255;0m";

// Slightly softer but still absolute (recommended for UI)
inline constexpr const char *red_soft = "\033[38;2;220;50;47m";
inline constexpr const char *green_soft = "\033[38;2;38;210;38m";  // alt green-ish
}  // namespace ansi

inline auto test_name(const std::string & name)
{
    std::cout << std::left << std::setw(width) << "    [TEST]: " + name;
}

inline auto passed()
{
    std::cout << ansi::green_soft << passed_str << ansi::reset << std::endl;
}

inline auto failed()
{
    std::cout << ansi::red_soft << failed_str << ansi::reset << std::endl;
}


// -----------------------------------------------------------------------------
// TESTS
// -----------------------------------------------------------------------------

void test_basic_scalars()
{
    test_name("Basic Scalars");
    cl::Parser p;
    auto i = p.add<cl::Num>(    cl::name("i", "int"), cl::desc("Integer"));
    auto f = p.add<cl::Fp_Num>( cl::name("f", "float"));
    auto s = p.add<cl::Text>(   cl::name("s", "string"));
    auto b = p.add<cl::Flag>(   cl::name("b", "bool"), cl::explicit_bool());
    auto neb = p.add<cl::Flag>( cl::name("B", "bo"));

    Args args = {"prog", "--bool=false", "--int", "42", "--float", "3.14159", "--string", "hello world", "--bo"};
    auto res = p.parse(args.argc(), args.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Num>(i) == 42);
        assert(std::abs(res->get<cl::Fp_Num>(f) - 3.14159) < 0.0001);
        assert(res->get<cl::Text>(s) == "hello world");
        assert(res->get<cl::Flag>(b) == false);
        assert(res->get<cl::Flag>(neb) == true);
        passed();
    }
    else
    {
        failed();
        std::cout << res.error() << std::endl;
    }

    test_name("Basic Scalars with '='");

    args = {"prog", "--int=42", "--float=-3.14159", "--string=hello world", "--bo"};

    res = p.parse(args.argc(), args.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Num>(i) == 42);
        assert(std::abs(res->get<cl::Fp_Num>(f) + 3.14159) < 0.0001);
        assert(res->get<cl::Text>(s) == "hello world");
        assert(res->get<cl::Flag>(b) == false);
        passed();
    }
    else
    {
        failed();
        std::cout << res.error() << std::endl;
    }
}

void test_bools()
{
    cl::Parser p;
    auto flag_simp = p.add<cl::Flag>(cl::name("s", "simple"));
    auto flag_expl = p.add<cl::Flag>(cl::name("e", "explicit"), cl::explicit_bool());
    auto flag_def  = p.add<cl::Flag>(cl::name("d", "default"), cl::explicit_bool(), cl::deflt(true));

    // ---------------------------------------------------------
    test_name("Boolean: Default States");
    Args a1 = {"prog"}; 
    auto res = p.parse(a1.argc(), a1.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Flag>(flag_simp) == false); // Default for flag is false
        assert(res->get<cl::Flag>(flag_expl) == false); // Default for bool is false
        assert(res->get<cl::Flag>(flag_def) == true);   // Explicit default
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Boolean: Implicit Flags (Presence = True)");
    Args a2 = {"prog", "-s"};
    res = p.parse(a2.argc(), a2.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Flag>(flag_simp) == true);
        // implicit flag cannot take value like -s=false (that would be parsed as -s, -f, -a...)
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Boolean: Explicit True Variants");
    // Testing: --explicit=true, --explicit yes, -e t
    Args a3 = {"prog", "--explicit=true", "--default", "yes"};
    res = p.parse(a3.argc(), a3.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Flag>(flag_expl) == true);
        assert(res->get<cl::Flag>(flag_def) == true);
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Boolean: Explicit False Variants");
    // Testing: --explicit=false, --explicit no, -e n
    Args a4 = {"prog", "--explicit=false", "--default", "no"};
    res = p.parse(a4.argc(), a4.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Flag>(flag_expl) == false);
        assert(res->get<cl::Flag>(flag_def) == false);
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Boolean: Short Option Explicit Assignment");
    // -e t (space), -d=f (equals)
    Args a5 = {"prog", "-e", "t", "-d=f"};
    res = p.parse(a5.argc(), a5.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Flag>(flag_expl) == true);  // t -> true
        assert(res->get<cl::Flag>(flag_def) == false);  // f -> false
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Boolean: Invalid Value Rejection");
    Args a6 = {"prog", "--explicit=maybe"};
    res = p.parse(a6.argc(), a6.ptr());

    if (!res.has_value())
    {
        // We expect failure here
        passed();
    }
    else
    {
        failed();
        std::cout << "Expected parse failure for invalid boolean string, but got success." << std::endl;
    }
}

void test_nums()
{
    std::cout << "\n\t **** Checking Nums\n" << std::endl;
    cl::Parser p;
    auto intf = p.add<cl::Num>   (cl::name("i", "int"));
    auto fltf = p.add<cl::Fp_Num>(cl::name("f", "flt"));

    auto dintf = p.add<cl::Num>   (cl::name("I", "dint"), cl::deflt(10));
    auto dfltf = p.add<cl::Fp_Num>(cl::name("F", "dflt"), cl::deflt(20.25));
    // ---------------------------------------------------------
    test_name("Checking defaults");
    Args a1 = {"prog"};
    auto res = p.parse(a1.argc(), a1.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Num>(dintf) == 10);
        assert(res->get<cl::Fp_Num>(dfltf) - 20.25 < 1e-6);
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    test_name("Checking next value short");
    a1 = {"prog", "-i", "1", "-f", "10.25", "-I", "-100", "-F", "-125.125"}; 
    res = p.parse(a1.argc(), a1.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Num>(dintf) == -100);
        assert(res->get<cl::Fp_Num>(dfltf) - (-125.125) < 1e-6);
        assert(res->get<cl::Num>(intf) == 1);
        assert(res->get<cl::Fp_Num>(fltf) - (10.25) < 1e-6);
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    test_name("Checking next value long");
    a1 = {"prog", "--int", "11", "--flt", "15.25", "--dint", "-0", "--dflt", "-125.125"}; 
    res = p.parse(a1.argc(), a1.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Num>(dintf) == 0);
        assert(res->get<cl::Fp_Num>(dfltf) - (-125.125) < 1e-6);
        assert(res->get<cl::Num>(intf) == 11);
        assert(res->get<cl::Fp_Num>(fltf) - (15.25) < 1e-6);
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }
}

template<typename T>
requires std::is_same_v<T, cl::Num>
struct Sq_check
{
    auto operator()(const T& val) const -> std::expected<void, std::string>
    {
        if (val*val == 225) return {};
        else return std::unexpected("BOOHOOOOOO!");
    }
    std::string help() const { return "^2 = 125"; }
};

void test_validator()
{
    std::cout << "\n\t **** Checking Validators\n" << std::endl;

    cl::Parser p("Hoiya", "Goiya");
    auto i = p.add<cl::Num>(cl::name("i", "int"), cl::validators(cl::Range<cl::Num>{10, 20}, Sq_check<cl::Num>{}));

    test_name("Passing validator");
    Args a = { "this", "-i15"};
    auto res = p.parse(a.argc(), a.ptr());
    if (res.has_value())
    {
        assert(res->get<cl::Num>(i) == 15);
        passed();
    }
    else
    {
        failed();
        std::cout << res.error() << std::endl;
    }
    test_name("Rejecting validator");
    a = { "this", "-i25"};
    res = p.parse(a.argc(), a.ptr());
    if (!res.has_value())
    {
        passed();
        std::cout << res.error();
    }
    else
    {
        failed();
    }
}

void test_short()
{
    std::cout << "\n\t **** Checking Shorts\n" << std::endl;
}


int main()
{
    std::cout << "**  TESTS BEGIN  **\n\n";
    try
    {
        test_basic_scalars();
        test_bools();
        test_nums();
        test_validator();
        std::cout << "\nAll tests passed successfully!\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test crashed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
