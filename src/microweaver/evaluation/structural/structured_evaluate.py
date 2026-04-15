import statistics

import os

from microweaver.evaluation.config import EvaluateConfig
from microweaver.util.file_op import load_json


class Evaluator:
    def __init__(self):
        self.name_service_map = None
        self.partitions = None
        self.data = load_json(EvaluateConfig.data_path)
        self.name_id_map = {item['name']: item['id'] for item in self.data}
        self.id_name_map = {item['id']: item['name'] for item in self.data}
        self.dependencies = {item['id']: item['dependencies'] for item in self.data}


    def calculate_SSB(self):
        """
        计算服务规模平衡性（Service Size Balance, SSB）
        公式：SSB = 标准差(服务节点数) / 均值(服务节点数)
        """
        # 统计每个服务的节点数
        service_sizes = [len(nodes) for nodes in self.partitions.values()]
        if len(service_sizes) <= 1:
            return 0.0  # 只有1个服务时无失衡问题

        # 计算均值和标准差
        mean_size = statistics.mean(service_sizes)
        std_size = statistics.stdev(service_sizes) if len(service_sizes) > 1 else 0.0

        # 计算SSB（避免除零错误）
        ssb = std_size / mean_size if mean_size > 0 else 0.0
        return ssb

    def calculate_SII(self):
        """
        计算结构不稳定性指数（Structural Instability Index, SII）
        公式：SII(S_k) = Fan_out / (Fan_in + Fan_out)
        Fan_out：服务对外调用数；Fan_in：服务被外部调用数
        返回：所有服务的平均SII
        """
        total_sii = 0.0
        num_services = 0

        # 预计算每个服务的Fan_in和Fan_out
        service_fan_out = {s: 0 for s in self.partitions.keys()}
        service_fan_in = {s: 0 for s in self.partitions.keys()}

        # 1. 计算Fan_out（服务对外调用数）
        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            fan_out = 0
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is None:
                    continue
                dep_node_ids = self.dependencies.get(node_id, [])
                for dep_node_id in dep_node_ids:
                    if dep_node_id not in service_node_ids:
                        dep_node_name = self.id_name_map.get(dep_node_id)
                        if dep_node_name:
                            dep_service = self.name_service_map.get(dep_node_name)
                            if dep_service:
                                fan_out += 1
            service_fan_out[service_name] = fan_out

        # 2. 计算Fan_in（服务被外部调用数）
        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            fan_in = 0
            # 遍历所有节点，统计调用当前服务的外部节点数
            for node_id, dep_node_ids in self.dependencies.items():
                node_name = self.id_name_map.get(node_id)
                if not node_name:
                    continue
                caller_service = self.name_service_map.get(node_name)
                if caller_service == service_name:
                    continue  # 内部调用，跳过
                # 检查是否调用当前服务的节点
                for dep_node_id in dep_node_ids:
                    if dep_node_id in service_node_ids:
                        fan_in += 1
            service_fan_in[service_name] = fan_in

        # 3. 计算每个服务的SII
        for service_name in self.partitions.keys():
            fan_in = service_fan_in[service_name]
            fan_out = service_fan_out[service_name]
            if fan_in + fan_out == 0:
                sii = 0.0
            else:
                sii = fan_out / (fan_in + fan_out) * len(self.partitions[service_name])
            total_sii += sii
            num_services += len(self.partitions[service_name])

        return total_sii / num_services if num_services > 0 else 0.0

    def calculate_modularity(self):
        """
        计算模块度（Modularity）
        将依赖图视为无向图，基于 Newman-Girvan 模块度定义：
        Q = Σ_k [ (l_k / m) - (d_k / (2m))^2 ]
        其中：
            - m：图中无向边的数量
            - l_k：社区（服务）k 内部的边数
            - d_k：社区（服务）k 中所有节点度数之和
        """
        # 构建 id -> service_name 映射
        id_service_map = {}
        for service_name, service_nodes in self.partitions.items():
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is not None:
                    id_service_map[node_id] = service_name

        # 统计总边数 m、每个节点度数以及每个服务内部边数
        m = 0  # 无向边数量（以依赖关系作无向边）
        node_degree = {node_id: 0 for node_id in self.dependencies.keys()}
        service_internal_edges = {service_name: 0 for service_name in self.partitions.keys()}

        for src_id, dep_ids in self.dependencies.items():
            for dst_id in dep_ids:
                m += 1
                # 将依赖视为无向边，出入度都 +1
                if src_id not in node_degree:
                    node_degree[src_id] = 0
                if dst_id not in node_degree:
                    node_degree[dst_id] = 0
                node_degree[src_id] += 1
                node_degree[dst_id] += 1

                src_service = id_service_map.get(src_id)
                dst_service = id_service_map.get(dst_id)
                if src_service is not None and src_service == dst_service:
                    service_internal_edges[src_service] = service_internal_edges.get(src_service, 0) + 1

        if m == 0:
            return 0.0

        # 计算每个服务的 d_k（社区内节点度数之和）
        service_degree_sum = {service_name: 0 for service_name in self.partitions.keys()}
        for node_id, degree in node_degree.items():
            service_name = id_service_map.get(node_id)
            if service_name is not None:
                service_degree_sum[service_name] = service_degree_sum.get(service_name, 0) + degree

        modularity = 0.0
        for service_name in self.partitions.keys():
            l_k = service_internal_edges.get(service_name, 0)
            d_k = service_degree_sum.get(service_name, 0)
            modularity += (l_k / m) - (d_k / (2 * m)) ** 2

        return modularity

    def calculate_ICP(self):
        """
        计算内部调用占比（Internal Call Proportion, ICP）
        公式：ICP = 内部调用数 / 总调用数
        返回：所有服务的平均ICP
        """
        internal_calls = 0
        total_calls = 0

        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}


            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is None:
                    continue
                dep_node_ids = self.dependencies.get(node_id, [])
                for dep_node_id in dep_node_ids:
                    total_calls += 1
                    if dep_node_id in service_node_ids:
                        internal_calls += 1

        return internal_calls / total_calls if total_calls > 0 else 0.0

    def evaluate(self):
        """
        执行评估，返回评估报告
        """
        results = []
        for folder in os.listdir(EvaluateConfig.partition_result_folder_path):
            dir_path = os.path.join(EvaluateConfig.partition_result_folder_path, folder)
            if os.path.isdir(dir_path):
                self.partitions = load_json(os.path.join(dir_path, "result.json"))
                self.name_service_map = {node: service for service in self.partitions for node in self.partitions[service]}

                report = {
                    "SSB": self.calculate_SSB(),
                    "SII": self.calculate_SII(),
                    "ICP": self.calculate_ICP(),
                    "Modularity": self.calculate_modularity(),
                }
                results.append(report)

        return results


def main():
    evaluator = Evaluator()
    evaluation_report = evaluator.evaluate()
    return evaluation_report
