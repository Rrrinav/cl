#include <print>
#include "cl.hpp"

int main(int argc, char *argv[])
{
    cl::Parser p("Hoi", "Heya mein friend.");

    auto a = p.add<cl::Num>( cl::name("a", "Ao"), cl::deflt(0), cl::multi());
    auto b = p.add<cl::Flag>(cl::name("b", "Bo"));
    auto c = p.add<cl::Fix_list<cl::Num, 3>>(cl::name("c", "Co"));
    auto d = p.add<cl::Fp_Num>(cl::name("d", "Do"), cl::env("c"));
    auto e = p.add<cl::Text>(cl::name("e", "Eo"));
    auto f = p.add<cl::Num>(cl::name("f", "Fo"));

    auto res = p.parse(argc, argv);
    if (!res)
    {
        std::println("{}", res.error());
        return 1;
    }
    std::println("a: {}", res->get<cl::List<cl::Num>>(a));
    std::println("b: {}", res->get<cl::Flag>(b));
    std::println("c: {}", res->get<cl::List<cl::Num>>(c));
    std::println("d: {}", res->get<cl::Fp_Num>(d));
    std::println("e: {}", res->get<cl::Text>(e));
    std::println("f: {}", res->get<cl::Num>(f));
}
