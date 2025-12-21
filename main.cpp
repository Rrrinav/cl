#include "cl.hpp"
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    // 1. Initialize Parser
    // Reserve space for options to avoid reallocations during setup
    cl::Parser parser("MyCLI", "A robust C++23 command line parser example", 20);

    // 2. Configure Parser (Optional)
    parser.cfg_.allow_combined_short_flags = true;          // -xvf
    parser.cfg_.allow_short_value_concat = true;            // -I/path
    parser.cfg_.value_binding = cl::P_SPACE | cl::P_EQUAL;  // --opt val or --opt=val

    auto verbose_id = parser.add(cl::Opt<bool>{{"verbose", "v"}, "Enable verbose logging"});

    auto count_id = parser.add(cl::Opt<int>{{"count", "c"}, "Number of iterations"}.deflt(1).validators(cl::Range<int>(1, 100)));

    // -- String Option with Default
    // Usage: --name "My Name", -nString
    auto name_id = parser.add(cl::Opt<std::string>{{"name", "n"}, "User name"}
            .deflt("Guest")
            .required()  // Mark as required (will fail if not present and no default is effectively used,
                         // though 'deflt' sets a value so 'required' is satisfied by the default technically
                         // in this specific logic, usually required means 'user must provide').
                         // In this lib, required checks 'seen', so if you provide a default,
                         // you usually don't mark required unless you want explicit user input.
    );

    // -- Float Array (Fixed Size)
    // Usage: --coords 1.0 2.0 3.0
    auto coords_id = parser.add(cl::Opt<std::array<double, 3>>{{"coords", ""}, "3D Coordinates (x y z)"}.deflt(0.0, 0.0, 0.0));

    // -- Multi-Value Integers (Vector)
    // Usage: --exclude 1 --exclude 2, or --exclude 1,2
    // Note: Internally stored as 'std::vector<long long>' (Canonical type for int vectors)
    auto exclude_id = parser.add(cl::Opt<int>{{"exclude", "x"}, "IDs to exclude"}.multi()  // Enable multiple occurrences
    );

    // 4. Parse Arguments
    auto result = parser.parse(argc, argv);

    // 5. Handle Errors
    if (!result)
    {
        std::cerr << "Error: " << result.error() << "\n\n";
        parser.print_help(std::cerr);
        return 1;
    }

    // 6. Retrieve Values
    // CRITICAL: We must retrieve using the CANONICAL types defined in cl.hpp
    // Int -> long long
    // Float -> double
    // String -> std::string
    // Bool -> bool

    bool verbose = parser.get<bool>(verbose_id);

    // Even though we passed cl::Opt<int>, the internal storage is long long.
    // We must request long long to avoid pointer casting errors.
    long long count = parser.get<long long>(count_id);
    long long count1 = parser.get<cl::Num>(count_id);

    const std::string &name = parser.get<std::string>(name_id);

    // Arrays are stored as std::array<Canonical, N>
    const auto &coords = parser.get<std::array<double, 3>>(coords_id);
    const auto &coords2 = parser.get<std::array<cl::Fp_Num, 3>>(coords_id);

    // Multi values are stored as std::vector<Canonical>
    const auto &excludes = parser.get<std::vector<long long>>(exclude_id);
    const auto &pos_args = parser.positionals();

    // 7. Use Values
    if (verbose)
        std::cout << "[Verbose Mode Enabled]\n";

    std::cout << "Name:   " << name << "\n";
    std::cout << "Count:  " << count << "\n";
    std::cout << "Coords: [" << coords[0] << ", " << coords[1] << ", " << coords[2] << "]\n";

    std::cout << "Excluded IDs: ";
    if (excludes.empty())
        std::cout << "(none)";
    for (auto id : excludes) std::cout << id << " ";
    std::cout << "\n";

    std::cout << "Positional Args: ";
    if (pos_args.empty())
        std::cout << "(none)";
    for (auto p : pos_args) std::cout << "'" << p << "' ";
    std::cout << "\n";

    return 0;
}
