#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Metrics {
    std::int64_t gid{};
    std::int64_t in_sum{};
    std::int64_t out_sum{};
    std::int64_t in_tx{};
    std::int64_t out_tx{};
    std::unordered_set<std::size_t> senders;
    std::unordered_set<std::size_t> recipients;
};

int main() {
    try {
        json input;
        std::cin >> input;
        const auto& input_nodes = input.at("nodes");
        const auto& input_edges = input.at("edges");

        std::vector<Metrics> nodes;
        nodes.reserve(input_nodes.size());
        std::unordered_map<std::int64_t, std::size_t> index;
        index.reserve(input_nodes.size() * 2);

        for (const auto& value : input_nodes) {
            const auto gid = value.is_object() ? value.at("gid").get<std::int64_t>()
                                               : value.get<std::int64_t>();
            if (index.find(gid) != index.end()) {
                throw std::runtime_error("duplicate gid: " + std::to_string(gid));
            }
            index.emplace(gid, nodes.size());
            nodes.push_back(Metrics{gid});
        }

        for (const auto& edge : input_edges) {
            const auto src_gid = edge.at("src").get<std::int64_t>();
            const auto dst_gid = edge.at("dst").get<std::int64_t>();
            // Money is transported and accumulated as integer tiyn (1/100 KZT).
            const auto amount = edge.at("sum_tiyn").get<std::int64_t>();
            const auto count = edge.at("n_tx").get<std::int64_t>();
            const auto src_it = index.find(src_gid);
            const auto dst_it = index.find(dst_gid);
            if (src_it == index.end() || dst_it == index.end()) {
                throw std::runtime_error("edge references unknown gid");
            }
            const auto src = src_it->second;
            const auto dst = dst_it->second;
            nodes[src].out_sum += amount;
            nodes[src].out_tx += count;
            nodes[src].recipients.insert(dst);
            nodes[dst].in_sum += amount;
            nodes[dst].in_tx += count;
            nodes[dst].senders.insert(src);
        }

        json output;
        output["engine"] = "cpp17";
        output["metrics"] = json::array();
        for (const auto& node : nodes) {
            const double pass_through = node.in_sum > 0
                ? static_cast<double>(node.out_sum) / static_cast<double>(node.in_sum) : 0.0;
            const double retention = node.in_sum > 0
                ? std::max(0.0, 1.0 - pass_through) : 0.0;
            output["metrics"].push_back({
                {"gid", node.gid},
                {"in_deg", node.senders.size()},
                {"out_deg", node.recipients.size()},
                {"unique_senders", node.senders.size()},
                {"unique_recipients", node.recipients.size()},
                {"in_kzt", static_cast<double>(node.in_sum) / 100.0},
                {"out_kzt", static_cast<double>(node.out_sum) / 100.0},
                {"in_tx", node.in_tx},
                {"out_tx", node.out_tx},
                {"pass_through", pass_through},
                {"retention_ratio", retention}
            });
        }
        std::cout << output.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        // stdout is deliberately reserved for valid JSON responses.
        std::cerr << "graph_core error: " << error.what() << '\n';
        return 1;
    }
}
