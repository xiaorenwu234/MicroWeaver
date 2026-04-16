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
        Calculate Service Size Balance (SSB)
        Formula: SSB = std_dev(service node counts) / mean(service node counts)
        """
        # Count nodes per service
        service_sizes = [len(nodes) for nodes in self.partitions.values()]
        if len(service_sizes) <= 1:
            return 0.0  # No imbalance issue when only 1 service

        # Calculate mean and std dev
        mean_size = statistics.mean(service_sizes)
        std_size = statistics.stdev(service_sizes) if len(service_sizes) > 1 else 0.0

        # Calculate SSB (avoid division by zero)
        ssb = std_size / mean_size if mean_size > 0 else 0.0
        return ssb

    def calculate_SII(self):
        """
        Calculate Structural Instability Index (SII)
        Formula: SII(S_k) = Fan_out / (Fan_in + Fan_out)
        Fan_out: number of outgoing calls from service; Fan_in: number of incoming calls to service
        Returns: average SII across all services
        """
        total_sii = 0.0
        num_services = 0

        # Pre-calculate Fan_in and Fan_out for each service
        service_fan_out = {s: 0 for s in self.partitions.keys()}
        service_fan_in = {s: 0 for s in self.partitions.keys()}

        # 1. Calculate Fan_out (outgoing calls from service)
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

        # 2. Calculate Fan_in (incoming calls to service)
        for service_name, service_nodes in self.partitions.items():
            service_node_ids = {self.name_id_map[node] for node in service_nodes if node in self.name_id_map}
            fan_in = 0
            # Iterate through all nodes, count external nodes calling current service
            for node_id, dep_node_ids in self.dependencies.items():
                node_name = self.id_name_map.get(node_id)
                if not node_name:
                    continue
                caller_service = self.name_service_map.get(node_name)
                if caller_service == service_name:
                    continue  # Internal call, skip
                # Check if calls nodes in current service
                for dep_node_id in dep_node_ids:
                    if dep_node_id in service_node_ids:
                        fan_in += 1
            service_fan_in[service_name] = fan_in

        # 3. Calculate SII for each service
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
        Calculate Modularity
        Treat dependency graph as undirected, based on Newman-Girvan modularity definition:
        Q = Σ_k [ (l_k / m) - (d_k / (2m))^2 ]
        Where:
            - m: number of undirected edges in graph
            - l_k: number of edges inside community (service) k
            - d_k: sum of degrees of all nodes in community (service) k
        """
        # Build id -> service_name mapping
        id_service_map = {}
        for service_name, service_nodes in self.partitions.items():
            for node in service_nodes:
                node_id = self.name_id_map.get(node)
                if node_id is not None:
                    id_service_map[node_id] = service_name

        # Count total edges m, degree per node, and internal edges per service
        m = 0  # Number of undirected edges (treat dependencies as undirected)
        node_degree = {node_id: 0 for node_id in self.dependencies.keys()}
        service_internal_edges = {service_name: 0 for service_name in self.partitions.keys()}

        for src_id, dep_ids in self.dependencies.items():
            for dst_id in dep_ids:
                m += 1
                # Treat dependency as undirected edge, increment both in and out degrees
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

        # Calculate d_k for each service (sum of degrees of nodes in community)
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
        Calculate Internal Call Proportion (ICP)
        Formula: ICP = internal_calls / total_calls
        Returns: average ICP across all services
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
        Execute evaluation, return evaluation report
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
