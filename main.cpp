#include <print>
#include <variant>
#include "cl.hpp"

int main(int argc, char *argv[])
{
    cl::Parser p("Hoi", "Heya mein friend.");

    auto a = p.add<cl::Num>(
        cl::name("a", "Ao"),
        cl::deflt(0),
        cl::multi()
    );
    auto b = p.add<cl::Flag>(cl::name("b", "Bo"));
    auto c = p.add<cl::Fix_list<cl::Num, 3>>(cl::name("c", "Co"));
    auto d = p.add<cl::Fp_Num>(cl::name("d", "Do"), cl::env("c"));
    auto e = p.add<cl::Text>(cl::name("e", "Eo"));
    auto f = p.add<cl::Num>(cl::name("f", "Fo"), cl::multi());

    auto res = p.parse(argc, argv);
    if (!res)
    {
        std::println("{}", res.error());
        return 1;
    }
    auto p_res = *res;
    std::println("a = {}", res->get<cl::List<cl::Num>>(a));
    auto c_vec = res->get<cl::List<cl::Num>>(c);
    std::println("c: {}", c_vec);
}
