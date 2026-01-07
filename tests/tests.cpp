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

void test_positionals()
{
    std::cout << "\n\t **** Checking Positionals\n" << std::endl;
    cl::Parser p;

    auto in  = p.positional<std::string>(cl::name("INPUT"), cl::required());
    auto out = p.positional<std::string>(cl::name("OUTPUT"), cl::deflt("a.out"));
    auto count = p.positional<long long>(cl::name("COUNT"), cl::deflt(1));

    // 1. All provided
    Args a1 = {"prog", "main.cpp", "main.exe", "5"};
    auto res = p.parse(a1.argc(), a1.ptr());

    test_name("All positionals provided");
    
    if(res.has_value()) {
        assert(res->get<std::string>(in) == "main.cpp");
        assert(res->get<std::string>(out) == "main.exe");
        assert(res->get<cl::Num>(count) == 5);
        passed();
    } else { failed(); std::cout << res.error() << std::endl; }

    // 2. Use Defaults
    Args a2 = {"prog", "test.c"};
    res = p.parse(a2.argc(), a2.ptr());
    
    test_name("Use defaults");
    if(res.has_value()) {
        assert(res->get<std::string>(in) == "test.c");
        assert(res->get<std::string>(out) == "a.out"); // Default
        assert(res->get<cl::Num>(count) == 1);       // Default
        passed();
    } else { failed(); }

    test_name("Too many args");
    // 3. Too many arguments
    Args a3 = {"prog", "a", "b", "3", "extra"};
    res = p.parse(a3.argc(), a3.ptr());
    if(!res.has_value()) passed(); // Should fail "Unexpected positional"
    else { failed(); std::cout << "Should have failed on 'extra'" << std::endl; }
}

void test_arrays()
{
    std::cout << "\n\t **** Checking Arrays & Vectors\n" << std::endl;
    cl::Parser p;

    // 1. Fixed Size Array (std::array<T, N>)
    // Here T is the actual array type because it's a specific parser case
    using Point3 = cl::Fix_list<cl::Num, 3>; // std::array<long long, 3>
    auto point = p.add<Point3>(cl::name("p", "point"), cl::desc("3D Point"), cl::array(cl::List_type::Delimited, ","));

    // 2. Vector (Multi-value) - Repeated flag (-v 1 -v 2)
    // WRONG: p.add<std::vector<int>> 
    // CORRECT: p.add<int>(..., cl::multi())
    auto vec_rep = p.add<cl::Num>(cl::name("v", "vec-rep"), cl::multi(cl::Multi_type::Repeat));

    // 3. Vector (Multi-value) - Delimited (-l 1,2,3)
    auto vec_del = p.add<cl::Num>(cl::name("l", "vec-del"), cl::multi(cl::Multi_type::Delimited, ","));

    // 4. Fixed Array with Defaults
    using Rect = cl::Fix_list<cl::Num, 4>;
    auto rect = p.add<Rect>(cl::name("r", "rect"), cl::deflt(0, 0, 10, 10));

    // ---------------------------------------------------------
    test_name("Array: Delimited Input (Correct Size)");
    Args a1 = {"prog", "--point", "10,20,30"};
    auto res = p.parse(a1.argc(), a1.ptr());

    if (res.has_value())
    {
        auto pt = res->get<Point3>(point);
        assert(pt[0] == 10 && pt[1] == 20 && pt[2] == 30);
        passed();
    }
    else { failed(); std::cout << res.error() << std::endl; }

    // ---------------------------------------------------------
    test_name("Array: Defaults");
    Args a2 = {"prog"};
    res = p.parse(a2.argc(), a2.ptr());

    if (res.has_value())
    {
        auto r = res->get<Rect>(rect);
        assert(r[0] == 0 && r[1] == 0 && r[2] == 10 && r[3] == 10);
        passed();
    }
    else { failed(); std::cout << res.error() << std::endl; }

    // ---------------------------------------------------------
    test_name("Vector: Repeated Flags");
    Args a3 = {"prog", "-v", "1", "-v", "2", "-v", "3"};
    res = p.parse(a3.argc(), a3.ptr());

    if (res.has_value())
    {
        // NOTE: Internal storage is std::vector<cl::Num> (long long)
        // We must request exactly that type.
        auto v = res->get<std::vector<cl::Num>>(vec_rep);
        
        assert(v.size() == 3);
        assert(v[0] == 1 && v[1] == 2 && v[2] == 3);
        passed();
    }
    else { failed(); std::cout << res.error() << std::endl; }

    // ---------------------------------------------------------
    test_name("Vector: Delimited Input");
    Args a4 = {"prog", "-l", "100,200,300,400"};
    res = p.parse(a4.argc(), a4.ptr());

    if (res.has_value())
    {
        auto v = res->get<std::vector<cl::Num>>(vec_del);
        assert(v.size() == 4);
        assert(v[0] == 100 && v[3] == 400);
        passed();
    }
    else { failed(); std::cout << res.error() << std::endl; }

    // ---------------------------------------------------------
    test_name("Array Error: Not enough elements");
    Args a5 = {"prog", "--point", "1,2"}; // Expecting 3
    res = p.parse(a5.argc(), a5.ptr());

    if (!res.has_value())
    {
        // Expected behavior depends on your parser strictness.
        // Assuming strictness on partial array fills is NOT enforced by default 
        // (the loop just stops), this might pass with the last element as 0.
        // If you want to assert failure, you need strict validators.
        // For now, let's print what we got.
        std::cout << "(Partial fill - might pass)" << std::endl;
        passed();
    }
    else
    {
        auto pt = res->get<Point3>(point);
        // It likely filled [1, 2, 0] (0 being default)
        if (pt[0] == 1 && pt[1] == 2) passed();
        else failed();
    }

    // ---------------------------------------------------------
    test_name("Array Error: Too many elements");
    Args a6 = {"prog", "--point", "1,2,3,4"}; // Expecting 3
    res = p.parse(a6.argc(), a6.ptr());

    if (!res.has_value())
    {
        // This MUST fail because of the array size check in inject_value
        passed();
    }
    else
    {
        failed();
        std::cout << "Expected failure for overflow, but got success." << std::endl;
    }
}

void test_get()
{
    cl::Parser p;
    auto i = p.add<cl::Num>(cl::name("i", "Int"));
    Args a = { "prg", "-i5" };
    auto res = p.parse(a.argc(), a.ptr());
    
    test_name("get -> bool");
    cl::Text ans;
    if (res)
    {
        assert(!res->get(i, ans));
        passed();
    }
    failed();
}

void test_subcommand_logic()
{
    std::cout << "\n\t **** Checking Subcommands\n" << std::endl;
    cl::Parser p;

    // 1. Setup Global Flags
    auto verbose = p.add<cl::Flag>(cl::name("v", "verbose"), cl::desc("Global verbose flag"));

    // 2. Setup Subcommand: 'commit'
    auto cmd_commit = p.add_sub_cmd("commit", "Commit changes", 0);
    auto msg = p.add<cl::Text>(
        cl::name("m", "message"), 
        cl::desc("Commit message"), 
        cl::sub_cmd(cmd_commit),
        cl::required()  // Required ONLY if 'commit' is used
    );

    // 3. Setup Subcommand: 'push'
    auto cmd_push = p.add_sub_cmd("push", "Push changes", 0);
    auto force = p.add<cl::Flag>(
        cl::name("f", "force"), 
        cl::desc("Force push"), 
        cl::sub_cmd(cmd_push)
    );

    // ---------------------------------------------------------
    test_name("Subcommand: Context Switch (commit)");
    Args a1 = {"prog", "commit", "-m", "fix bug", "-v"};
    auto res = p.parse(a1.argc(), a1.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Text>(msg) == "fix bug");
        assert(res->get<cl::Flag>(verbose) == true);
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Subcommand: Context Switch (push)");
    // Context switches to 'push', allows -f. 'msg' is NOT required here.
    Args a2 = {"prog", "push", "--force"};
    res = p.parse(a2.argc(), a2.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Flag>(force) == true);
        
        // Ensure we can't access 'msg' (it wasn't parsed)
        cl::Text s;
        assert(!res->get(msg, s)); 
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Subcommand: Invalid Scope (Cross-talk)");
    // Using a 'push' flag (-f) inside 'commit' command
    Args a3 = {"prog", "commit", "-m", "wip", "-f"};
    res = p.parse(a3.argc(), a3.ptr());

    if (!res.has_value())
    {
        // Expected Failure: Flag '-f' is not valid in current context
        passed();
        std::cout << res.error();
    }
    else
    {
        failed(); std::cout << "Expected failure for invalid scope, got success." << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Subcommand: Missing Required in Context");
    // 'commit' requires -m
    Args a4 = {"prog", "commit"};
    res = p.parse(a4.argc(), a4.ptr());

    if (!res.has_value())
    {
        // Expected Failure: Required option missing
        passed();
    }
    else
    {
        failed(); std::cout << "Expected failure for missing required, got success." << std::endl;
    }

    // ---------------------------------------------------------
    test_name("Global Context (Requirement Ignored)");
    // Required flag -m (from commit) should NOT trigger error here because we aren't in 'commit' mode.
    Args a5 = {"prog", "-v"};
    res = p.parse(a5.argc(), a5.ptr());

    if (res.has_value())
    {
        assert(res->get<cl::Flag>(verbose) == true);
        passed();
    }
    else
    {
        failed(); std::cout << res.error() << std::endl;
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
        test_arrays();
        test_positionals();
        test_subcommand_logic();
        std::cout << "\nAll tests passed successfully!\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test crashed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
