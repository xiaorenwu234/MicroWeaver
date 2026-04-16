import requests
import json

import networkx as nx
from datetime import datetime, timedelta
from sqlglot import parse, exp

from microweaver.input_builder.config import InputConfig


def queryGraphql(query, variables):
    url = "http://localhost:8080/graphql"
    headers = {
        "Content-Type": "application/json",
    }

    # Send POST request
    response = requests.post(url, json={"query": query, "variables": variables}, headers=headers)

    # Check response
    if response.status_code == 200:
        data = response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

    return data

def queryServices():
    # Define GraphQL query
    query = """
    query queryServices($layer: String!) {
        services: listServices(layer: $layer) {
            id
            value: name
            label: name
            group
            layers
            normal
            shortName
        }
    }
    """

    variables = {
        "layer":"GENERAL"
    }

    data = queryGraphql(query, variables)

    services = data.get("data").get("services")
    # print("services:")
    # print(json.dumps(services, indent=1))
    return services

def queryTraces(serviceId):
    # Define GraphQL query
    query = """
    query queryTraces($condition: TraceQueryCondition) {
        data: queryBasicTraces(condition: $condition) {
            traces {
                key: segmentId
                endpointNames
                duration
                start
                isError
                traceIds
            }
        }
    }
    """
    
    time_gap = timedelta(hours=1)
    current_time = datetime.now() - timedelta(hours=8)
    former_time = current_time - time_gap
    
    variables = {
        "condition": {
            "queryDuration": {
                "start": former_time.strftime("%Y-%m-%d %H%M"),
                "end": current_time.strftime("%Y-%m-%d %H%M"),
                "step": "MINUTE"
            },
            "traceState": "ALL",
            "queryOrder": "BY_START_TIME",
            "paging": {
                "pageNum": 1,
                "pageSize": 100
            },
            "minTraceDuration": None,
            "maxTraceDuration": None,
            "serviceId": serviceId
        }
    }

    data = queryGraphql(query, variables)

    traces = data.get("data").get("data").get("traces")
    return traces

def queryTrace(traceId):
    # Define GraphQL query
    query = """
    query queryTrace($traceId: ID!) {
        trace: queryTrace(traceId: $traceId) {
            spans {
                traceId
                segmentId
                spanId
                parentSpanId
                refs {
                    traceId
                    parentSegmentId
                    parentSpanId
                    type
                }
                serviceCode
                serviceInstanceName
                startTime
                endTime
                endpointName
                type
                peer
                component
                isError
                layer
                tags {
                    key
                    value
                }
                logs {
                    time
                    data {
                        key
                        value
                    }
                }
                attachedEvents {
                    startTime {
                        seconds
                        nanos
                    }
                    event
                    endTime {
                        seconds
                        nanos
                    }
                    tags {
                        key
                        value
                    }
                    summary {
                        key
                        value
                    }
                }
            }
        }
    }
    """
    
    variables = {
        "traceId": traceId
    }

    data = queryGraphql(query, variables)

    spans = data.get("data").get("trace").get("spans")
    return spans


def getAttributes(span):
    excluded_components = ["HikariCP"]
    if span["component"] in excluded_components:
        return None
    
    if span["type"] == "Entry":
        return None
    endpointNames = [span["endpointName"]]
    type = "function"
    if span["layer"] == "Database":
        type = "database"
        tags = {tag["key"]: tag["value"] for tag in span["tags"]}
        sql = tags["db.statement"]
        if not sql:
            return None 
        # print(json.dumps(span, indent=1))
        tables = []
        try:
            parsed_sqls = parse(sql)
        except Exception as e:
            print(e)
            return None
        for parsed_sql in parsed_sqls:
            tables += [table.this.sql() for table in parsed_sql.find_all(exp.Table)]
        if not tables:
            return None
        endpointNames = [f'{tags["db.type"]}.{tags["db.instance"]}.{table}' for table in tables]
        
    keys = ["traceId", "segmentId", "spanId", "parentSpanId", "startTime", "endTime"]
    nodes = []
    for endpointName in endpointNames:
        attributes = {}
        attributes["endpointName"] = endpointName
        attributes["type"] = type
        for key in keys:
            attributes[key] = span[key]
        nodes += [attributes]
    return nodes


def getNodeId(node):
    node_id = node["endpointName"]
    if node_id.endswith(")"):
        node_id = ".".join(node_id.split("(")[0].split(".")[:-1])
    return node_id


def addToGraph(G, spans):
    weight = 1
    for span in spans:
        nodes = getAttributes(span)
        if not nodes:
            continue
        
        for node in nodes:
            node_id = getNodeId(node)
            execution_time = node["endTime"] - node["startTime"]
            
            if G.has_node(node_id):
                G.nodes[node_id]["execution_time"] += execution_time
                G.nodes[node_id]["count"] += 1
            else:
                G.add_node(node_id, execution_time=execution_time, count=1, type=node["type"])

            # If parentSpanId is not 0, add edge
            if node["parentSpanId"] != 0:
                parent_node = next((n for n in spans if n["spanId"] == node["parentSpanId"]), None)
                if parent_node:
                    parent_node_id = getNodeId(parent_node)
                    if G.has_edge(parent_node_id, node_id):
                        G[parent_node_id][node_id]["weight"] += weight
                    else:
                        G.add_edge(parent_node_id, node_id, weight=weight)
            else:
                G.nodes[node_id]["type"] = "entry"
                

def staticGraph(G):
    with open("callGraph.json", "r") as infile:
        callGraph = json.load(infile)
    print(callGraph)
    for caller, callees in callGraph.items():
        G.add_node(caller)
        for callee in callees:
            if G.has_edge(caller, callee):
                G[caller][callee]["weight"] += 1
            else:
                G.add_edge(caller, callee, weight=1)
    return G


def buildGraph(G=nx.DiGraph(), source='json', saveSpans=True, proj_name=None):
    if source == 'json':
        with open(f"spans_{proj_name}.json", "r") as infile:
            spanss = json.load(infile)
    elif source == 'static':
        return staticGraph(G)
    else:
        spanss = []
        services = queryServices()
        # print(json.dumps(services, indent=1))
        for service in services:
            traces = queryTraces(service["id"]) 
            # print(json.dumps(traces, indent=1))
            for trace in traces:
                spans = queryTrace(trace["traceIds"][0])
                spanss.append(spans)

    for spans in spanss:
        addToGraph(G, spans)
    
    return G


def writeDependencies(G, config: InputConfig):
    """
    Output all nodes in graph G in dependency format
    """
    # First create mapping from node_id to id
    node_to_id = {}
    for idx, node_id in enumerate(G.nodes()):
        node_to_id[node_id] = idx
    
    result = []
    
    for node_id in G.nodes():
        # Get node dependencies (outgoing edge target nodes) and convert to corresponding ids
        dependencies = [node_to_id[target] for source, target in G.out_edges(node_id)]
        
        # Build node information
        node_info = {
            "id": len(result),  # Use current list length as ID
            "name": node_id.split(".")[-1],  # Use last part of qualifiedName as name
            "qualifiedName": node_id,  # Use node_id as qualifiedName
            "dependencies": dependencies
        }
        
        result.append(node_info)
    
    with open(config.dynamic_json_path, "w") as f:
        json.dump(result, f, indent=4)
    
    return result

def main(config: InputConfig):
    G = buildGraph(source='j', saveSpans=True, proj_name="test")
    writeDependencies(G, config)