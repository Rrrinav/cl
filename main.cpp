#include <cstdint>
#include <print>
#include "cl.hpp"

int main(int argc, char *argv[])
{
    cl::Parser p("", "");

    auto d_s = p.add_sub_cmd("device", "device configurations", 0);
    auto s_s = p.add_sub_cmd("software", "software management", 0);
    auto u_s = p.add_sub_cmd("update", "update software", 0, s_s);

    auto d_a = p.add<cl::Text>(cl::name("a", "add"), cl::desc("device to add"), cl::sub_cmd(d_s));
    auto s_a = p.add<cl::Text>(cl::name("a", "add"), cl::desc("software to add"), cl::sub_cmd(s_s));
    auto u_a = p.add<cl::Text>(cl::name("a", "add"), cl::desc("software to add"), cl::sub_cmd(u_s));
    auto S_a = p.add<cl::Text>(cl::name("a", "add"), cl::desc("something to add"));

    auto res = p.parse(argc, argv);

    if (!res)
        std::println("{}", res.error());
    std::string add_text = "";

    if (res->get<cl::Text>(d_a, add_text))
        std::println("device add: {}", add_text);
    if (res->get<cl::Text>(s_a, add_text))
        std::println("software add: {}", add_text);

    auto l = res->get<cl::Text>("add");
    std::println("l: {}", l.value_or("hui-pui"));

    std::println("add nested: {}", res->get_subcmd("software")->get_subcmd("update")->get<cl::Text>("add").value_or("omg chain"));
    std::println("add but nested in single call: {}", res->get<cl::Text>("software", "update", "add").value_or("omg chain"));

    //auto d_res = res->get_sub(d_res);
    //d_res->get<cl::Num>(d_a);
    //res->get<cl::Num>("device", "add");
    //res->get<cl::Num>("add");
}
