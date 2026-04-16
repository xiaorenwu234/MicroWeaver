#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microservice Split Visualization V2 - Two-level Interactive Visualization
Level 1: Microservice-level Overview (Bubbles + Service Dependencies)
Level 2: Click to view internal class nodes of microservice
"""

import json
import math
import os

from microweaver.visualization.config import VisualizationConfig
COLORS = [
    "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4",
    "#10b981", "#f97316", "#8b5cf6", "#ec4899", "#14b8a6",
    "#e11d48", "#0ea5e9", "#d97706", "#84cc16"
]

def main(config: VisualizationConfig):
    print("Loading data...")
    with open(config.data_path, "r", encoding="utf-8") as f:
        data_nodes = json.load(f)
    with open(config.result_path, "r", encoding="utf-8") as f:
        ms_map = json.load(f)

    print(f"Total nodes: {len(data_nodes)}, Microservices: {len(ms_map)}")

    # Build mapping (support both name and filePath matching)
    name_to_node = {}
    filepath_to_node = {}
    id_to_node = {}
    for node in data_nodes:
        name_to_node[node["name"]] = node
        if node.get("filePath"):
            filepath_to_node[node["filePath"]] = node
        id_to_node[node["id"]] = node

    node_to_ms = {}
    ms_nodes = {}
    for ms_name, members in ms_map.items():
        ms_nodes[ms_name] = []
        for member in members:
            # Priority: match by class name, then by filePath
            matched_node = name_to_node.get(member) or filepath_to_node.get(member)
            if matched_node:
                nid = matched_node["id"]
                node_to_ms[nid] = ms_name
                ms_nodes[ms_name].append(nid)

    # Sort by node count
    ms_list = sorted(ms_nodes.keys(), key=lambda k: len(ms_nodes[k]), reverse=True)

    # Count edges
    intra_edges = {}   # ms_name -> [(src, tgt), ...]
    inter_edges = {}   # (ms_a, ms_b) -> count
    inter_edge_details = {}  # (ms_a, ms_b) -> [(src_id, tgt_id), ...]

    for node in data_nodes:
        src_id = node["id"]
        if src_id not in node_to_ms:
            continue
        src_ms = node_to_ms[src_id]
        for tgt_id in node["dependencies"]:
            if tgt_id not in node_to_ms:
                continue
            tgt_ms = node_to_ms[tgt_id]
            if src_ms == tgt_ms:
                if src_ms not in intra_edges:
                    intra_edges[src_ms] = []
                intra_edges[src_ms].append((src_id, tgt_id))
            else:
                key = tuple(sorted([src_ms, tgt_ms]))
                inter_edges[key] = inter_edges.get(key, 0) + 1
                if key not in inter_edge_details:
                    inter_edge_details[key] = []
                inter_edge_details[key].append((src_id, tgt_id))

    # ===== Build Level 1 Data: Microservice Level =====
    # One node per microservice
    ms_overview_nodes = []
    for i, ms_name in enumerate(ms_list):
        display = ms_name if ms_name else "(Unnamed)"
        count = len(ms_nodes[ms_name])
        ms_overview_nodes.append({
            "id": ms_name,
            "name": display,
            "count": count,
            "color": COLORS[i % len(COLORS)],
            "index": i
        })

    ms_overview_links = []
    for (a, b), cnt in sorted(inter_edges.items(), key=lambda x: -x[1]):
        ms_overview_links.append({
            "source": a,
            "target": b,
            "count": cnt
        })

    # ===== Build Level 2 Data: Inside Each Microservice =====
    ms_detail_data = {}
    for ms_name in ms_list:
        nids = ms_nodes[ms_name]
        nodes_data = []
        nid_set = set(nids)
        
        # Calculate degree for each node (used for node size)
        node_degree = {}
        edges_in = intra_edges.get(ms_name, [])
        for s, t in edges_in:
            node_degree[s] = node_degree.get(s, 0) + 1
            node_degree[t] = node_degree.get(t, 0) + 1

        # Find which nodes have cross-service dependencies
        cross_nodes = set()
        for (a, b), details in inter_edge_details.items():
            if ms_name in (a, b):
                for s, t in details:
                    if s in nid_set:
                        cross_nodes.add(s)
                    if t in nid_set:
                        cross_nodes.add(t)

        for nid in nids:
            node = id_to_node[nid]
            deg = node_degree.get(nid, 0)
            nodes_data.append({
                "id": str(nid),
                "name": node["name"],
                "qualifiedName": node.get("qualifiedName", ""),
                "degree": deg,
                "isCross": nid in cross_nodes
            })

        edges_data = []
        for s, t in edges_in:
            edges_data.append({"source": str(s), "target": str(t)})

        # Cross-service edges (related to this microservice)
        cross_edges = []
        for (a, b), details in inter_edge_details.items():
            if ms_name in (a, b):
                other = b if a == ms_name else a
                for s, t in details:
                    if s in nid_set or t in nid_set:
                        cross_edges.append({
                            "source": str(s),
                            "target": str(t),
                            "otherService": other if other else "(Unnamed)",
                            "internal": s in nid_set and t in nid_set
                        })

        ms_detail_data[ms_name] = {
            "nodes": nodes_data,
            "intraEdges": edges_data,
            "crossEdges": cross_edges
        }

    # ===== Generate HTML =====
    print("Generating HTML...")

    overview_nodes_json = json.dumps(ms_overview_nodes, ensure_ascii=False)
    overview_links_json = json.dumps(ms_overview_links, ensure_ascii=False)
    detail_json = json.dumps(ms_detail_data, ensure_ascii=False)
    ms_list_json = json.dumps(ms_list, ensure_ascii=False)
    colors_json = json.dumps(COLORS)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Microservice Architecture Visualization</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background: #ffffff;
  font-family: 'Segoe UI', -apple-system, sans-serif;
  color: #333333;
  overflow: hidden;
  height: 100vh;
  user-select: none;
}}
#header {{
  position: fixed; top:0; left:0; right:0; z-index:100;
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid rgba(0,0,0,0.08);
  padding: 10px 24px;
  display: flex; align-items: center; justify-content: space-between;
  height: 48px;
}}
#header h1 {{
  font-size: 17px; font-weight: 600;
  background: linear-gradient(90deg, #6366f1, #22c55e, #f59e0b);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.header-info {{ font-size: 12px; color: #888; }}
#breadcrumb {{
  position: fixed; top: 48px; left: 0; right: 0; z-index: 99;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(0,0,0,0.06);
  padding: 6px 24px;
  font-size: 13px; color: #666;
  display: flex; align-items: center; gap: 8px;
  height: 32px;
}}
#breadcrumb a {{
  color: #6366f1; cursor: pointer; text-decoration: none;
}}
#breadcrumb a:hover {{ text-decoration: underline; }}
#breadcrumb .sep {{ color: #bbb; }}
#breadcrumb .current {{ color: #333; }}
#canvas-container {{
  position: absolute; top: 80px; left: 0; right: 0; bottom: 0;
}}
svg {{ width: 100%; height: 100%; }}

/* Side info panel */
#info-panel {{
  position: fixed; top: 80px; right: 0; bottom: 0; width: 300px;
  background: rgba(255, 255, 255, 0.97);
  backdrop-filter: blur(16px);
  border-left: 1px solid rgba(0,0,0,0.08);
  z-index: 98; overflow-y: auto; padding: 16px;
  transform: translateX(100%);
  transition: transform 0.3s ease;
}}
#info-panel.open {{ transform: translateX(0); }}
#info-panel h3 {{
  font-size: 13px; color: #555; text-transform: uppercase;
  letter-spacing: 1px; margin-bottom: 12px;
}}
.info-section {{
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(0,0,0,0.06);
}}
.info-label {{ font-size: 11px; color: #888; margin-bottom: 2px; }}
.info-value {{ font-size: 14px; color: #333; }}
.dep-item {{
  display: flex; justify-content: space-between;
  padding: 4px 8px; margin: 2px 0;
  background: rgba(0,0,0,0.03);
  border-radius: 4px; font-size: 12px;
}}
.dep-item .dep-name {{ color: #555; }}
.dep-item .dep-count {{ color: #ef4444; font-weight: 600; }}

/* Legend */
#legend {{
  position: fixed; bottom: 16px; left: 16px; z-index: 98;
  background: rgba(255, 255, 255, 0.93);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 10px; padding: 12px 16px;
  max-height: 300px; overflow-y: auto;
}}
.legend-item {{
  display: flex; align-items: center; gap: 8px;
  padding: 3px 0; font-size: 12px; color: #555;
  cursor: pointer;
}}
.legend-item:hover {{ color: #000; }}
.legend-dot {{
  width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
}}
.legend-count {{
  font-size: 10px; color: #999; margin-left: auto;
}}

/* Tooltip */
.tooltip {{
  position: absolute; pointer-events: none; z-index: 200;
  background: rgba(255, 255, 255, 0.97);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(0,0,0,0.1);
  border-radius: 8px; padding: 10px 14px;
  font-size: 12px; color: #555;
  box-shadow: 0 4px 20px rgba(0,0,0,0.12);
  max-width: 360px;
  display: none;
}}
.tooltip .tt-title {{ font-size: 14px; font-weight: 600; color: #222; margin-bottom: 4px; }}
.tooltip .tt-sub {{ color: #888; font-size: 11px; }}

/* Search box */
#search-box {{
  position: fixed; top: 86px; left: 50%; transform: translateX(-50%);
  z-index: 100; display: none;
}}
#search-input {{
  background: rgba(255,255,255,0.97);
  border: 1px solid rgba(0,0,0,0.15);
  border-radius: 8px; padding: 8px 16px;
  font-size: 13px; color: #333; width: 320px;
  outline: none;
}}
#search-input:focus {{ border-color: #6366f1; }}
#search-results {{
  background: rgba(255,255,255,0.98);
  border: 1px solid rgba(0,0,0,0.08);
  border-top: none; border-radius: 0 0 8px 8px;
  max-height: 200px; overflow-y: auto;
}}
.search-item {{
  padding: 6px 16px; font-size: 12px; color: #555;
  cursor: pointer;
}}
.search-item:hover {{ background: rgba(99,102,241,0.1); color: #222; }}

/* Buttons */
.ctrl-btn {{
  position: fixed; z-index: 100;
  background: rgba(99,102,241,0.1);
  border: 1px solid rgba(99,102,241,0.3);
  color: #6366f1; padding: 6px 14px; border-radius: 6px;
  cursor: pointer; font-size: 12px;
  transition: all 0.2s;
}}
.ctrl-btn:hover {{ background: rgba(99,102,241,0.2); color: #4f46e5; }}
#btn-search {{ top: 86px; right: 16px; }}
#btn-info {{ top: 86px; right: 100px; }}
#btn-back {{ top: 86px; left: 16px; display: none; }}
</style>
</head>
<body>
<div id="header">
  <h1>Microservice Architecture Visualization</h1>
  <div class="header-info">{sum(len(v) for v in ms_nodes.values())} classes · {len(ms_list)} microservices</div>
</div>
<div id="breadcrumb">
  <a id="bc-home" onclick="goOverview()">Overview</a>
</div>

<button class="ctrl-btn" id="btn-back" onclick="goOverview()">← Back</button>
<button class="ctrl-btn" id="btn-info" onclick="toggleInfo()">ℹ Info</button>
<button class="ctrl-btn" id="btn-search" onclick="toggleSearch()">🔍 Search</button>

<div id="search-box">
  <input id="search-input" placeholder="Search class name..." oninput="onSearch(this.value)">
  <div id="search-results"></div>
</div>

<div id="canvas-container">
  <svg id="svg"></svg>
</div>

<div class="tooltip" id="tooltip"></div>
<div id="info-panel"></div>
<div id="legend"></div>

<script>
// ===== DATA =====
const msNodes = {overview_nodes_json};
const msLinks = {overview_links_json};
const detailData = {detail_json};
const msList = {ms_list_json};
const colors = {colors_json};

const tooltip = d3.select('#tooltip');
const svg = d3.select('#svg');
const container = document.getElementById('canvas-container');
let currentView = 'overview'; // 'overview' or service name
let simulation = null;

// ===== OVERVIEW =====
function renderOverview() {{
  currentView = 'overview';
  document.getElementById('btn-back').style.display = 'none';
  document.getElementById('breadcrumb').innerHTML = '<span class="current">Overview · Microservice Architecture Overview</span>';
  document.getElementById('info-panel').classList.remove('open');
  document.getElementById('search-box').style.display = 'none';

  svg.selectAll('*').remove();
  if (simulation) simulation.stop();

  const W = container.clientWidth;
  const H = container.clientHeight;

  const g = svg.append('g');

  // Zoom
  const zoom = d3.zoom()
    .scaleExtent([0.3, 5])
    .on('zoom', (e) => g.attr('transform', e.transform));
  svg.call(zoom);
  svg.call(zoom.transform, d3.zoomIdentity.translate(W/2, H/2).scale(0.9));

  // Node size mapping
  const maxCount = d3.max(msNodes, d => d.count);
  const rScale = d3.scaleSqrt().domain([1, maxCount]).range([30, 140]);

  // Edge width mapping
  const maxEdge = d3.max(msLinks, d => d.count) || 1;
  const wScale = d3.scaleLinear().domain([1, maxEdge]).range([1, 12]);

  // Create node and id mapping
  const nodeMap = new Map();
  const simNodes = msNodes.map(d => {{
    const obj = {{...d, r: rScale(d.count)}};
    nodeMap.set(d.id, obj);
    return obj;
  }});

  const simLinks = msLinks.map(d => ({{
    source: d.source, target: d.target, count: d.count
  }}));

  // Force simulation
  simulation = d3.forceSimulation(simNodes)
    .force('charge', d3.forceManyBody().strength(-800))
    .force('center', d3.forceCenter(0, 0))
    .force('collision', d3.forceCollide(d => d.r + 30))
    .force('link', d3.forceLink(simLinks).id(d => d.id).distance(350).strength(0.3))
    .on('tick', ticked);

  // Draw edges
  const linkG = g.append('g');
  const links = linkG.selectAll('line')
    .data(simLinks).enter().append('line')
    .attr('stroke', 'rgba(239,68,68,0.35)')
    .attr('stroke-width', d => wScale(d.count))
    .attr('stroke-linecap', 'round');

  // Numbers on edges
  const linkLabels = linkG.selectAll('text')
    .data(simLinks).enter().append('text')
    .text(d => d.count)
    .attr('font-size', 10)
    .attr('fill', 'rgba(239,68,68,0.6)')
    .attr('text-anchor', 'middle')
    .attr('dy', -4);

  // Draw nodes
  const nodeG = g.append('g');
  const nodes = nodeG.selectAll('g')
    .data(simNodes).enter().append('g')
    .attr('cursor', 'pointer')
    .call(d3.drag()
      .on('start', dragStart)
      .on('drag', dragging)
      .on('end', dragEnd))
    .on('click', (e, d) => goDetail(d.id))
    .on('mouseover', (e, d) => showTooltip(e, d))
    .on('mouseout', () => tooltip.style('display', 'none'));

  // Outer glow
  nodes.append('circle')
    .attr('r', d => d.r + 6)
    .attr('fill', 'none')
    .attr('stroke', d => d.color)
    .attr('stroke-width', 2)
    .attr('stroke-opacity', 0.15);

  // Main circle
  nodes.append('circle')
    .attr('r', d => d.r)
    .attr('fill', d => d.color)
    .attr('fill-opacity', 0.12)
    .attr('stroke', d => d.color)
    .attr('stroke-width', 2)
    .attr('stroke-opacity', 0.6);

  // Name
  nodes.append('text')
    .text(d => d.name)
    .attr('text-anchor', 'middle')
    .attr('dy', -8)
    .attr('font-size', d => Math.max(11, Math.min(16, d.r / 5)))
    .attr('fill', d => d.color)
    .attr('font-weight', 600);

  // Count
  nodes.append('text')
    .text(d => d.count + ' classes')
    .attr('text-anchor', 'middle')
    .attr('dy', 12)
    .attr('font-size', 11)
    .attr('fill', '#999');

  function ticked() {{
    links
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    linkLabels
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2);
    nodes.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
  }}

  function dragStart(e, d) {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
  function dragging(e, d) {{ d.fx = e.x; d.fy = e.y; }}
  function dragEnd(e, d) {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}

  function showTooltip(e, d) {{
    // Find cross-service dependencies related to this service
    let deps = msLinks.filter(l => l.source.id === d.id || l.target.id === d.id || l.source === d.id || l.target === d.id);
    let depsHtml = deps.map(l => {{
      let other = (l.source.id || l.source) === d.id ? (l.target.name || l.target) : (l.source.name || l.source);
      return `<div style="font-size:11px;color:#666">${{other}}: <span style="color:#ef4444">${{l.count}}</span></div>`;
    }}).join('');
    tooltip.html(`
      <div class="tt-title">${{d.name}}</div>
      <div class="tt-sub">${{d.count}} classes</div>
      ${{depsHtml ? '<div style="margin-top:6px;border-top:1px solid rgba(0,0,0,0.06);padding-top:6px">Cross-service deps:</div>' + depsHtml : ''}}
    `).style('display', 'block')
      .style('left', (e.pageX + 16) + 'px')
      .style('top', (e.pageY - 10) + 'px');
  }}

  // Legend
  renderLegend();
}}

// ===== DETAIL VIEW =====
function goDetail(msName) {{
  currentView = msName;
  const msIdx = msList.indexOf(msName);
  const color = colors[msIdx % colors.length];
  const displayName = msName || '(Unnamed)';
  const detail = detailData[msName];
  if (!detail) return;

  document.getElementById('btn-back').style.display = 'block';
  document.getElementById('breadcrumb').innerHTML =
    `<a onclick="goOverview()">Overview</a><span class="sep">›</span><span class="current" style="color:${{color}}">${{displayName}}</span><span style="color:#555;margin-left:8px">${{detail.nodes.length}} classes, ${{detail.intraEdges.length}} internal deps</span>`;

  svg.selectAll('*').remove();
  if (simulation) simulation.stop();
  tooltip.style('display', 'none');

  const W = container.clientWidth;
  const H = container.clientHeight;
  const g = svg.append('g');

  const zoom = d3.zoom().scaleExtent([0.1, 8])
    .on('zoom', (e) => g.attr('transform', e.transform));
  svg.call(zoom);

  // Node size based on degree
  const maxDeg = d3.max(detail.nodes, d => d.degree) || 1;
  const rScale = d3.scaleLinear().domain([0, maxDeg]).range([3, 14]);

  const simNodes = detail.nodes.map(d => ({{...d, r: rScale(d.degree)}}));
  const simLinks = detail.intraEdges.map(d => ({{source: d.source, target: d.target}}));

  // Adjust force parameters based on node count
  const n = simNodes.length;
  const chargeStrength = n > 1000 ? -15 : n > 500 ? -30 : -60;
  const distFn = n > 1000 ? 20 : n > 500 ? 35 : 50;

  simulation = d3.forceSimulation(simNodes)
    .force('charge', d3.forceManyBody().strength(chargeStrength))
    .force('center', d3.forceCenter(0, 0))
    .force('collision', d3.forceCollide(d => d.r + 2))
    .force('link', d3.forceLink(simLinks).id(d => d.id).distance(distFn).strength(0.2))
    .alphaDecay(0.02)
    .on('tick', ticked);

  // Initial zoom
  const initScale = Math.min(W, H) / (Math.sqrt(n) * 25);
  svg.call(zoom.transform, d3.zoomIdentity.translate(W/2, H/2).scale(Math.min(2, Math.max(0.15, initScale))));

  // Edges
  const linkG = g.append('g');
  const links = linkG.selectAll('line')
    .data(simLinks).enter().append('line')
    .attr('stroke', color)
    .attr('stroke-opacity', 0.15)
    .attr('stroke-width', 0.8);

  // Nodes
  const nodeG = g.append('g');
  const nodes = nodeG.selectAll('circle')
    .data(simNodes).enter().append('circle')
    .attr('r', d => d.r)
    .attr('fill', d => d.isCross ? '#ef4444' : color)
    .attr('fill-opacity', d => d.isCross ? 0.8 : 0.6)
    .attr('stroke', d => d.isCross ? '#ef4444' : color)
    .attr('stroke-width', d => d.isCross ? 1.5 : 0.5)
    .attr('stroke-opacity', 0.8)
    .on('mouseover', (e, d) => {{
      d3.select(e.target).attr('r', d.r * 2).attr('fill-opacity', 1);
      tooltip.html(`
        <div class="tt-title">${{d.name}}</div>
        <div class="tt-sub">${{d.qualifiedName}}</div>
        <div style="margin-top:4px;font-size:11px;color:#666">Degree: ${{d.degree}}${{d.isCross ? ' · <span style=color:#ef4444>Has cross-service deps</span>' : ''}}</div>
      `).style('display', 'block')
        .style('left', (e.pageX + 12) + 'px')
        .style('top', (e.pageY - 10) + 'px');
    }})
    .on('mouseout', (e, d) => {{
      d3.select(e.target).attr('r', d.r).attr('fill-opacity', d.isCross ? 0.8 : 0.6);
      tooltip.style('display', 'none');
    }});

  // Labels - only show for high-degree nodes
  const labelThreshold = n > 500 ? Math.max(5, maxDeg * 0.3) : Math.max(2, maxDeg * 0.15);
  const labels = nodeG.selectAll('text')
    .data(simNodes.filter(d => d.degree >= labelThreshold))
    .enter().append('text')
    .text(d => d.name)
    .attr('font-size', 9)
    .attr('fill', '#555')
    .attr('dx', d => d.r + 3)
    .attr('dy', 3);

  function ticked() {{
    links
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodes.attr('cx', d => d.x).attr('cy', d => d.y);
    labels.attr('x', d => d.x).attr('y', d => d.y);
  }}

  // Side panel - show cross-service dependency statistics
  showDetailInfo(msName, displayName, color, detail);
}}

function showDetailInfo(msName, displayName, color, detail) {{
  const panel = document.getElementById('info-panel');
  // Count cross-service dependencies
  const crossMap = {{}};
  detail.crossEdges.forEach(e => {{
    crossMap[e.otherService] = (crossMap[e.otherService] || 0) + 1;
  }});
  const crossArr = Object.entries(crossMap).sort((a,b) => b[1] - a[1]);
  
  panel.innerHTML = `
    <h3 style="color:${{color}}">${{displayName}}</h3>
    <div class="info-section">
      <div class="info-label">Classes</div>
      <div class="info-value">${{detail.nodes.length}}</div>
    </div>
    <div class="info-section">
      <div class="info-label">Internal Dependencies</div>
      <div class="info-value">${{detail.intraEdges.length}}</div>
    </div>
    <div class="info-section">
      <div class="info-label">Cross-service Dependencies</div>
      <div class="info-value">${{detail.crossEdges.length}}</div>
    </div>
    ${{crossArr.length ? '<div class="info-section"><div class="info-label">Depends on</div>' +
      crossArr.map(([name, cnt]) =>
        `<div class="dep-item"><span class="dep-name">${{name}}</span><span class="dep-count">${{cnt}}</span></div>`
      ).join('') + '</div>' : ''}}
    <div style="margin-top:12px;font-size:11px;color:#999">
      <span style="color:#ef4444">\u25cf</span> Red nodes = cross-service boundary<br>
      Hover nodes for details
    </div>
  `;
  panel.classList.add('open');
}}

function goOverview() {{
  renderOverview();
}}

function renderLegend() {{
  const legend = document.getElementById('legend');
  legend.innerHTML = '<div style="font-size:11px;color:#999;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px">Microservices</div>' +
    msNodes.map(d =>
      `<div class="legend-item" onclick="goDetail('${{d.id}}')">
        <div class="legend-dot" style="background:${{d.color}}"></div>
        <span>${{d.name}}</span>
        <span class="legend-count">${{d.count}}</span>
      </div>`
    ).join('') +
    '<div style="margin-top:10px;border-top:1px solid rgba(0,0,0,0.06);padding-top:8px">' +
    '<div style="font-size:11px;color:#999">Click bubble to drill down</div>' +
    '<div style="font-size:11px;color:#999">Drag to reposition</div>' +
    '<div style="font-size:11px;color:#999">Scroll to zoom</div></div>';
}}

function toggleInfo() {{
  document.getElementById('info-panel').classList.toggle('open');
}}

function toggleSearch() {{
  const box = document.getElementById('search-box');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
  if (box.style.display === 'block') document.getElementById('search-input').focus();
}}

function onSearch(val) {{
  const results = document.getElementById('search-results');
  if (val.length < 2) {{ results.innerHTML = ''; return; }}
  val = val.toLowerCase();
  let found = [];
  for (const msName of msList) {{
    const detail = detailData[msName];
    for (const n of detail.nodes) {{
      if (n.name.toLowerCase().includes(val) || n.qualifiedName.toLowerCase().includes(val)) {{
        found.push({{ name: n.name, ms: msName || '(Unnamed)', msId: msName }});
        if (found.length >= 20) break;
      }}
    }}
    if (found.length >= 20) break;
  }}
  results.innerHTML = found.map(f =>
    `<div class="search-item" onclick="goDetail('${{f.msId}}')">${{f.name}} <span style="color:#555">· ${{f.ms}}</span></div>`
  ).join('');
}}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') {{
    if (currentView !== 'overview') goOverview();
    document.getElementById('search-box').style.display = 'none';
    document.getElementById('info-panel').classList.remove('open');
  }}
  if (e.key === '/' && e.target.tagName !== 'INPUT') {{
    e.preventDefault();
    toggleSearch();
  }}
}});

// Start
renderOverview();
window.addEventListener('resize', () => {{
  if (currentView === 'overview') renderOverview();
}});
</script>
</body>
</html>'''

    with open(config.html_save_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(config.html_save_path) / 1024 / 1024
    print(f"\n✅ Generated: {os.path.abspath(config.html_save_path)} ({size_mb:.1f} MB)")
    print(f"   Microservices: {len(ms_list)}")
    print(f"   Total nodes: {sum(len(ms_nodes[ms]) for ms in ms_list)}")


if __name__ == "__main__":
    main()
