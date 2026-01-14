#include <print>

#define CL_IMPLEMENTATION
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
    auto arr = p.add<cl::Fix_list<cl::Num, 4>>(cl::name("h", "hadd"));

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

    auto val = res->get_subcmd("software")->get_subcmd("update")->get<cl::Text>("add").value_or("omg chain");
    std::println("add nested: {}", val);

    val = res->get<cl::Text>("software", "update", "add").value_or("omg chain");
    std::println("add nested but nested in single call: {}", val);

    std::string val_try;
    if (auto resu = res->get<cl::Text>(&val_try, "software", "update", "add"); resu)
        std::println("add nested but nested in single call: {}", val_try);
    else
        std::println("{}", resu.error());


    std::println("was seen: {}", res->is_seen(S_a));
    std::println("arr: {}", res->get<cl::Fix_list<cl::Num, 4>>(arr).value_or({6,7,8,9}));
}
