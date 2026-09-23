#!/usr/bin/env python3
"""
Стартовый код кейса «Граф денег» — HackAlem AI.

Что он делает:
  1. грузит три parquet-файла и проверяет их консистентность;
  2. собирает направленный взвешенный граф;
  3. считает БАЗОВЫЕ метрики узлов (степени, обороты, PageRank);
  4. пишет три выгрузки в требуемой ТЗ схеме — с ПУСТЫМИ ролями.

MVP также присваивает объяснимые роли, строит Louvain-кластеры,
ранжирует узлы и запускает локальный UI поиска по gid.

Запуск:
    python starter.py
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import networkx as nx

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


# ---------------------------------------------------------------- загрузка

def load(data_dir: Path):
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def sanity_check(edges, nodes, tx):
    """Проверки, которые стоит пройти до того, как строить модель."""
    print("=" * 64)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 64)
    print(f"  узлов в nodes.parquet : {len(nodes):>6}")
    print(f"  рёбер                 : {len(edges):>6}")
    print(f"  транзакций            : {len(tx):>6}")
    print(f"  seed-клиентов         : {int(nodes.is_seed.sum()):>6}")
    print(f"  оборот, KZT           : {edges.sum_kzt.sum():>14,.0f}")
    print(f"  период                : {tx.date.min().date()} — {tx.date.max().date()}")

    # транзакции должны складываться в рёбра
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    assert (m._merge == "both").all(), "edges и transactions не сходятся по парам"
    print("  edges == transactions : OK")

    # узлы без единого ребра
    in_edges = set(edges.src) | set(edges.dst)
    orphans = set(nodes.gid) - in_edges
    print(f"\n  ВНИМАНИЕ: {len(orphans)} узлов нет ни в одном ребре "
          f"(из них seed: {len(orphans & set(nodes[nodes.is_seed].gid))})")
    print("  → они всё равно должны попасть в nodes_roles.csv")
    print("=" * 64, "\n")
    return orphans


# ---------------------------------------------------------------- граф

def build_graph(edges, nodes=None) -> nx.DiGraph:
    """Направленный граф. sum_kzt — вес ребра, n_tx — количество переводов."""
    G = nx.DiGraph()
    if nodes is not None:
        G.add_nodes_from(nodes.gid.astype(int))
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def basic_features(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Базовые метрики. Это старт, а не финиш — добавляйте свои."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_kzt = dict(G.in_degree(weight="sum_kzt"))
    out_kzt = dict(G.out_degree(weight="sum_kzt"))
    in_tx = dict(G.in_degree(weight="n_tx"))
    out_tx = dict(G.out_degree(weight="n_tx"))
    pr = nx.pagerank(G, weight="sum_kzt")

    df = nodes[["gid", "depth", "is_seed"]].copy()
    df["in_deg"] = df.gid.map(in_deg).fillna(0).astype(int)
    df["out_deg"] = df.gid.map(out_deg).fillna(0).astype(int)
    df["in_kzt"] = df.gid.map(in_kzt).fillna(0.0)
    df["out_kzt"] = df.gid.map(out_kzt).fillna(0.0)
    df["in_tx"] = df.gid.map(in_tx).fillna(0).astype(int)
    df["out_tx"] = df.gid.map(out_tx).fillna(0).astype(int)
    df["pagerank"] = df.gid.map(pr).fillna(0.0)

    # доля полученного, которая ушла дальше. Около 1.0 — деньги не задерживаются.
    df["pass_through"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.replace(0, np.nan), np.nan)
    # A numeric zero keeps the required CSV field filled; evidence still shows in_kzt=0.
    df["pass_through"] = df["pass_through"].fillna(0.0)

    # ЛОВУШКА КЕЙСА: узел на 4-м колене без исходящих может быть не «стоком»,
    # а просто местом, где закончился обход. Разберитесь с этим.
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    return df


def cpp_basic_features(G: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame,
                       core_path: Path) -> pd.DataFrame:
    """Get linear graph aggregates from the C++17 core over stdin/stdout JSON."""
    core_path = core_path.resolve()
    if not core_path.is_file():
        raise FileNotFoundError(
            f"C++ core not found: {core_path}\n"
            "Build it first: cmake -S cpp_core -B cpp_core/build -A x64 && "
            "cmake --build cpp_core/build --config Release\n"
            "Emergency fallback: add --python-metrics")
    payload = {
        "nodes": [int(x) for x in nodes.gid],
        "edges": [{"src": int(r.src), "dst": int(r.dst),
                   "sum_tiyn": int(round(r.sum_kzt * 100)), "n_tx": int(r.n_tx)}
                  for r in edges.itertuples(index=False)]
    }
    try:
        process = subprocess.run(
            [str(core_path)], input=json.dumps(payload, separators=(",", ":")),
            text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"C++ core failed (exit {exc.returncode}): {exc.stderr.strip()}") from exc
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("C++ core returned invalid JSON; stdout must contain JSON only") from exc
    if result.get("engine") != "cpp17":
        raise RuntimeError("Unexpected C++ core response: missing engine=cpp17")
    metrics = pd.DataFrame(result["metrics"])
    if len(metrics) != len(nodes) or metrics.gid.nunique() != len(nodes):
        raise RuntimeError(f"C++ core returned {len(metrics)} rows for {len(nodes)} nodes")
    metrics[["in_kzt", "out_kzt"]] = metrics[["in_kzt", "out_kzt"]].astype(float)
    df = nodes[["gid", "depth", "is_seed"]].merge(metrics, on="gid", how="left", validate="one_to_one")
    pr = nx.pagerank(G, weight="sum_kzt")
    df["pagerank"] = df.gid.map(pr).fillna(0.0)
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    print(f"Metrics engine: C++17 ({core_path})")
    return df


def enrich(G: nx.DiGraph, df: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, explainable roles, clusters and priority scores."""
    df = df.copy()
    active = pd.concat([tx[["src", "date"]].rename(columns={"src": "gid"}),
                        tx[["dst", "date"]].rename(columns={"dst": "gid"})])
    df["active_days"] = df.gid.map(active.groupby("gid").date.nunique()).fillna(0).astype(int)
    communities = nx.community.louvain_communities(G.to_undirected(), weight="sum_kzt", seed=42)
    communities = sorted(communities, key=lambda c: (-len(c), min(c)))
    df["cluster_id"] = df.gid.map({g: i for i, c in enumerate(communities) for g in c}).astype(int)
    between = nx.betweenness_centrality(G, k=min(300, len(G)), normalized=True, seed=42)
    df["betweenness"] = df.gid.map(between).fillna(0.0)

    # Explainable context signals: graph distance to any seed and relative community size.
    seeds = [int(g) for g in df.loc[df.is_seed, "gid"]]
    seed_distance = nx.multi_source_dijkstra_path_length(G.to_undirected(), seeds, weight=None)
    df["seed_proximity"] = df.gid.map(lambda g: 1.0 / (1.0 + seed_distance.get(int(g), 99)))
    cluster_sizes = df.groupby("cluster_id").size()
    cluster_significance = cluster_sizes.rank(method="average", pct=True)
    df["cluster_significance"] = df.cluster_id.map(cluster_significance).astype(float)

    pct = lambda s: s.rank(method="average", pct=True).fillna(0.0)
    pin, pout = pct(df.in_deg), pct(df.out_deg)
    pvol = pct(np.log1p(df.in_kzt + df.out_kzt))
    ppr, pbet = pct(df.pagerank), pct(df.betweenness)
    roles, scores = [], []
    for i, r in df.iterrows():
        if r.in_deg == 0 and r.out_deg == 0:
            role, score = "peripheral", .35
        elif r.out_deg == 0:
            role = "peripheral" if r.truncated_by_depth else "terminal"
            score = .35 + .25 * pin[i] if role == "peripheral" else .45 + .45 * pin[i]
        elif r.in_deg >= 2 and pin[i] >= .75 and r.in_deg > 1.25 * r.out_deg:
            role, score = "consolidator", .45 + .45 * pin[i]
        elif r.out_deg >= 3 and pout[i] >= .75 and r.out_deg > 1.25 * r.in_deg:
            role, score = "distributor", .45 + .45 * pout[i]
        elif r.in_deg > 0 and r.out_deg > 0 and .65 <= r.pass_through <= 1.35:
            role, score = "transit", .50 + .30 * (1 - min(abs(r.pass_through - 1), 1)) + .15 * pbet[i]
        elif r.in_deg > 0 and r.out_deg > 0 and (pbet[i] >= .8 or (pin[i] >= .65 and pout[i] >= .65)):
            role, score = "coordinator", .45 + .25 * pbet[i] + .15 * max(pin[i], pout[i])
        else:
            role, score = "peripheral", .35 + .30 * pvol[i]
        roles.append(role); scores.append(min(float(score), 1.0))
    df["role"], df["role_score"] = roles, scores
    structural = np.maximum.reduce([pin.to_numpy(), pout.to_numpy(), ppr.to_numpy(), pbet.to_numpy()])
    role_weight = df.role.map({"coordinator": 1.0, "consolidator": .9, "distributor": .9,
                               "transit": .7, "terminal": .45, "peripheral": .15})
    df["priority_score"] = np.clip(
        .35 * ppr + .18 * pbet + .14 * pvol + .08 * structural +
        .10 * role_weight + .08 * df.seed_proximity + .07 * df.cluster_significance, 0, 1)

    def explain(r):
        facts = (f"вход: {r.in_deg} контр., {r.in_tx} пер., {r.in_kzt:.0f} KZT; "
                 f"выход: {r.out_deg} контр., {r.out_tx} пер., {r.out_kzt:.0f} KZT")
        if r.truncated_by_depth:
            return f"Периферийный узел: {facts}. Граница depth=4 — конечный статус не подтверждён."
        hypotheses = {
            "consolidator": "Признаки аккумуляции средств",
            "transit": f"Признаки транзита, доля дальнейшего перевода {r.pass_through:.2f}",
            "distributor": "Признаки распределения средств",
            "terminal": "Вероятный конечный получатель в наблюдаемом графе",
            "coordinator": "Структурно значимый посредник, рекомендуется проверка",
            "peripheral": "Слабо выраженная роль в наблюдаемом графе",
        }
        return f"{hypotheses[r.role]}: {facts}."
    df["evidence"] = df.apply(explain, axis=1).str.slice(0, 200)
    return df


# ---------------------------------------------------------------- выгрузки

def write_outputs(df: pd.DataFrame, out_dir: Path, edges=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Обязательная схема фиксирована ТЗ; расширенные метрики остаются только в df/UI.
    roles = df[["gid", "role", "role_score", "cluster_id",
                "priority_score", "evidence"]].copy()
    roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    cmap = df.set_index("gid").cluster_id
    ec = edges.assign(cs=edges.src.map(cmap), cd=edges.dst.map(cmap))
    internal = ec[ec.cs == ec.cd].groupby("cs").sum_kzt.sum()
    clusters = []
    for cid, group in df.groupby("cluster_id", sort=True):
        top = group.nlargest(5, "priority_score"); counts = group.role.value_counts(); dominant = counts.index[0]
        n_nodes, n_seed = len(group), int(group.is_seed.sum())
        internal_kzt = float(internal.get(cid, 0))
        hypothesis = (f"Доминирующая роль — {dominant} ({counts[dominant]} из {n_nodes}); "
                      f"размер {n_nodes}, seed {n_seed}, внутренний оборот {internal_kzt:.0f} KZT. "
                      "Структура может отражать связанную группу переводов и требует контекстной проверки.")
        clusters.append({"cluster_id": cid, "n_nodes": n_nodes, "n_seed": n_seed,
                         "sum_kzt_internal": internal_kzt,
                         "top_gids": "|".join(top.gid.astype(str)),
                         "hypothesis": hypothesis})
    pd.DataFrame(clusters).to_csv(out_dir / "clusters.csv", index=False)

    top = df.nlargest(50, "priority_score").reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1)); top["why"] = top.evidence
    top[["rank", "gid", "role", "priority_score", "why"]].to_csv(out_dir / "top_nodes.csv", index=False)

    print(f"Выгрузки записаны в {out_dir}/")


# ---------------------------------------------------------------- подсказки

def hints(G: nx.DiGraph, df: pd.DataFrame):
    """Куда смотреть дальше. Ответов здесь нет — только направления."""
    print("\nС ЧЕГО НАЧАТЬ")
    print("-" * 64)
    print(f"  узлов, получающих от 3+ разных плательщиков : {(df.in_deg >= 3).sum()}")
    print(f"  узлов, рассылающих на 10+ получателей       : {(df.out_deg >= 10).sum()}")
    print(f"  узлов и с входом, и с выходом               : {((df.in_deg > 0) & (df.out_deg > 0)).sum()}")
    print(f"  узлов, обрезанных 4-м коленом               : {df.truncated_by_depth.sum()}  <- разберитесь")
    print(f"  слабосвязных компонент                      : {nx.number_weakly_connected_components(G)}")
    print("""
  Вопросы, на которые стоит ответить метриками:
    * чем «деньги пришли и остались» отличается от «пришли и ушли дальше»?
    * что важнее для роли — количество плательщиков или сумма?
    * узел собирает средства от нескольких SEED — это случайность или структура?
    * если убрать узел, сеть распадётся или переживёт?

  Полезное в networkx: pagerank, hits, betweenness_centrality,
  community.louvain_communities, simple_cycles, all_simple_paths.
  Не забудьте: граф НАПРАВЛЕННЫЙ и ВЗВЕШЕННЫЙ.
""")


HTML = '''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Money Graph AML</title>
<style>:root{--bg:#07111f;--panel:#101d2e;--line:#24354b;--text:#e8eef7;--muted:#91a4bd;--accent:#4f8cff}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#112b4a 0,var(--bg) 36%);color:var(--text);font:14px Inter,Segoe UI,system-ui,sans-serif}.shell{max-width:1440px;margin:auto;padding:20px}.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}.brand{font-size:22px;font-weight:750}.brand span{color:#63a2ff}.badge{padding:6px 10px;border:1px solid #284667;border-radius:20px;color:#9ec5ff}.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}.kpi,.toolbar,.panel{background:rgba(16,29,46,.96);border:1px solid #20344c;border-radius:14px;box-shadow:0 15px 45px #02081766}.kpi{padding:12px 14px}.kpi b{display:block;font-size:20px;color:#ddebff}.kpi span{color:var(--muted);font-size:12px}.toolbar{display:flex;gap:9px;padding:12px;margin-bottom:14px;flex-wrap:wrap}input,select,button{height:38px;border-radius:8px;border:1px solid #30465f;background:#0a1626;color:var(--text);padding:0 12px}input{min-width:265px;flex:1}button{border:0;background:linear-gradient(135deg,#3478ef,#625cff);font-weight:700;cursor:pointer}.search-notice{display:none;margin:-5px 0 14px;padding:9px 11px;background:#3b2b13;border:1px solid #715322;border-radius:8px;color:#ffd999}.grid{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px}.panel{overflow:hidden}.panel-title{padding:13px 16px;border-bottom:1px solid var(--line);font-weight:700}.graph-wrap{position:relative;min-height:590px}.graph-wrap svg{width:100%;height:590px;display:block}.empty{position:absolute;inset:0;display:grid;place-items:center;color:var(--muted);pointer-events:none}.notice{display:none;margin:10px 14px 0;padding:9px 11px;background:#3b2b13;border:1px solid #715322;border-radius:8px;color:#ffd999}.card{padding:15px;border-bottom:1px solid var(--line)}.gid{font-size:17px;font-weight:750;word-break:break-all}.meta{display:flex;gap:7px;flex-wrap:wrap;margin:9px 0}.pill{padding:4px 8px;border-radius:15px;background:#172a41;color:#bcd1e8}.evidence{color:#c5d3e4;line-height:1.45}.top{max-height:420px;overflow:auto}.top-row{display:grid;grid-template-columns:30px 1fr auto;gap:8px;padding:9px 14px;border-bottom:1px solid #1b2c41;cursor:pointer}.top-row:hover,.top-row.active{background:#162a42}.top-gid{font-family:Consolas,monospace;font-size:12px}.score{color:#79e6b2}.legend{display:flex;gap:10px;flex-wrap:wrap;padding:12px 15px;border-top:1px solid var(--line)}.legend span{color:var(--muted);font-size:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.edge{stroke:#506984;stroke-width:1.35;opacity:.72}.label{fill:#b7c8da;font:10px Consolas,monospace;paint-order:stroke;stroke:#07111f;stroke-width:3px}.node{stroke:#dceaff;stroke-width:1.2;cursor:pointer}.selected{stroke:#fff;stroke-width:3;filter:drop-shadow(0 0 7px #fff)}@media(max-width:950px){.kpis{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.graph-wrap,.graph-wrap svg{height:480px;min-height:480px}.top{max-height:240px}}</style></head>
<body><div class="shell"><div class="header"><div class="brand">Money Graph <span>AML Dashboard</span></div><div class="badge">C++17 metrics · Python analytics</div></div>
<div id="kpis" class="kpis"></div>
<div class="toolbar"><input id="gid" placeholder="Введите полный gid" onkeydown="if(event.key==='Enter')go()"><button onclick="go()">Найти узел</button><select id="roleFilter" onchange="applyFilters()"><option value="">Все роли</option></select><select id="clusterFilter" onchange="applyFilters()"><option value="">Все кластеры</option></select><select id="colorMode" onchange="renderLegend();applyFilters()"><option value="role">Цвет: роль</option><option value="cluster">Цвет: кластер</option></select><button onclick="resetFilters()">Сбросить фильтры</button></div>
<div id="searchNotice" class="search-notice" role="status"></div>
<div class="grid"><section class="panel"><div class="panel-title">Локальная сеть переводов</div><div id="limitNotice" class="notice"></div><div class="graph-wrap"><div id="empty" class="empty">Найдите узел или выберите его в топ-листе</div><svg id="graph" viewBox="0 0 960 590"></svg></div><div id="legend" class="legend"></div></section>
<aside class="panel"><div class="panel-title">Карточка узла</div><div id="card" class="card"><div class="evidence">Узел не выбран</div></div><div class="panel-title">Топ-20 приоритетов</div><div id="top" class="top"></div></aside></div></div>
<script>
const colors={consolidator:'#f59e0b',transit:'#22d3ee',distributor:'#8b5cf6',terminal:'#ef4444',coordinator:'#22c55e',peripheral:'#64748b'};let current=null;
const $=id=>document.getElementById(id),gid=$('gid'),roleFilter=$('roleFilter'),clusterFilter=$('clusterFilter'),colorMode=$('colorMode'),legend=$('legend'),topList=$('top'),card=$('card'),graph=$('graph'),empty=$('empty'),limitNotice=$('limitNotice'),searchNotice=$('searchNotice'),kpis=$('kpis');
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function init(){roleFilter.innerHTML+=Object.keys(colors).map(r=>`<option value="${r}">${r}</option>`).join('');renderLegend();try{await Promise.all([loadKpis(),loadTop()])}catch(e){showNotice('Не удалось загрузить начальные данные: '+e.message)}}
async function getJson(url){let r=await fetch(url);if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json()}
async function loadKpis(){let d=await getJson('/api/kpi');let items=[['Узлы',d.nodes.toLocaleString('ru-RU')],['Рёбра',d.edges.toLocaleString('ru-RU')],['Кластеры',d.clusters.toLocaleString('ru-RU')],['Оборот',Math.round(d.turnover_kzt).toLocaleString('ru-RU')+' KZT'],['Расчёт',d.timing_ms.toLocaleString('ru-RU')+' мс']];kpis.innerHTML=items.map(x=>`<div class="kpi"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('')}
async function loadTop(){let d=await getJson('/api/top');if(!Array.isArray(d)||!d.length)throw new Error('top_nodes.csv пуст');topList.innerHTML=d.map(x=>`<div class="top-row" data-gid="${x.gid}" onclick="selectGid('${x.gid}')"><span>${x.rank}</span><span><div class="top-gid">${x.gid}</div><small style="color:${colors[x.role]}">${x.role} · C${x.cluster_id}</small></span><b class="score">${(+x.priority_score).toFixed(3)}</b></div>`).join('');await selectGid(d[0].gid)}
function selectGid(v){gid.value=v;return go()}
async function go(){let id=gid.value.trim();if(!id){showNotice('Введите gid. Текущий граф сохранён.');return}try{let d=await getJson('/api/node?gid='+encodeURIComponent(id));if(d.error){showNotice('Узел '+id+' не найден. Текущий граф сохранён.');return}searchNotice.style.display='none';current=d;empty.style.display='none';renderCard(d.node);fillClusters(d.nodes);applyFilters();document.querySelectorAll('.top-row').forEach(x=>x.classList.toggle('active',x.dataset.gid===String(d.node.gid)));limitNotice.style.display=d.limited?'block':'none';limitNotice.textContent=d.limited?`Показаны 80 из ${d.total_edges} связей. Визуализация ограничена для быстрого отображения.`:''}catch(e){showNotice('Ошибка загрузки узла: '+e.message+'. Текущий граф сохранён.')}}
function showNotice(message){searchNotice.textContent=message;searchNotice.style.display='block'}
function renderCard(n){card.innerHTML=`<div class="gid">${n.gid}</div><div class="meta"><span class="pill" style="color:${colors[n.role]}">${n.role}</span><span class="pill">cluster ${n.cluster_id}</span><span class="pill">priority ${(+n.priority_score).toFixed(3)}</span></div><div class="evidence">${esc(n.evidence)}</div><div class="meta"><span class="pill">in ${n.in_deg}</span><span class="pill">out ${n.out_deg}</span><span class="pill">depth ${n.depth}</span></div>`}
function fillClusters(ns){let old=clusterFilter.value,vals=[...new Set(ns.map(n=>n.cluster_id))].sort((a,b)=>a-b);clusterFilter.innerHTML='<option value="">Все кластеры</option>'+vals.map(c=>`<option value="${c}">Кластер ${c}</option>`).join('');if(vals.map(String).includes(old))clusterFilter.value=old}
function resetFilters(){roleFilter.value='';clusterFilter.value='';applyFilters()}
function applyFilters(){if(!current)return;let role=roleFilter.value,cluster=clusterFilter.value;let ns=current.nodes.filter(n=>(!role||n.role===role)&&(!cluster||String(n.cluster_id)===cluster));if(!ns.some(n=>String(n.gid)===String(current.node.gid))&&(!role||current.node.role===role)&&(!cluster||String(current.node.cluster_id)===cluster))ns.unshift(current.node);let ids=new Set(ns.map(n=>String(n.gid))),es=current.edges.filter(e=>ids.has(String(e.src))&&ids.has(String(e.dst)));draw(ns,es,current.node.gid)}
function clusterColor(cid){let hue=(Number(cid)*137.508+18)%360;return `hsl(${hue.toFixed(1)} 72% 58%)`}
function nodeColor(n){return colorMode.value==='cluster'?clusterColor(n.cluster_id):colors[n.role]}
function renderLegend(){if(colorMode.value==='cluster'&&current){let cs=[...new Set(current.nodes.map(n=>n.cluster_id))].sort((a,b)=>a-b);legend.innerHTML=cs.map(c=>`<span><i class="dot" style="background:${clusterColor(c)}"></i>кластер ${c}</span>`).join('')}else{legend.innerHTML=Object.entries(colors).map(([r,c])=>`<span><i class="dot" style="background:${c}"></i>${r}</span>`).join('')}}
function draw(ns,es,centerId){let cx=480,cy=295,R=Math.min(225,150+ns.length*2),pos={};let center=ns.findIndex(n=>String(n.gid)===String(centerId));if(center>0)[ns[0],ns[center]]=[ns[center],ns[0]];ns.forEach((n,i)=>pos[n.gid]=i?{x:cx+R*Math.cos(2*Math.PI*(i-1)/Math.max(1,ns.length-1)),y:cy+R*Math.sin(2*Math.PI*(i-1)/Math.max(1,ns.length-1))}:{x:cx,y:cy});let out='<defs><marker id="arrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#7891ac"/></marker></defs>';es.forEach(e=>{let a=pos[e.src],b=pos[e.dst];out+=`<line class="edge" marker-end="url(#arrow)" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"><title>${e.src} → ${e.dst}: ${e.sum_kzt} KZT, ${e.n_tx} переводов</title></line>`});ns.forEach(n=>{let p=pos[n.gid],sel=String(n.gid)===String(centerId)?' selected':'';out+=`<circle class="node${sel}" fill="${nodeColor(n)}" cx="${p.x}" cy="${p.y}" r="${sel?11:8}" onclick="selectGid('${n.gid}')"><title>${n.gid} · ${n.role} · cluster ${n.cluster_id}</title></circle><text class="label" x="${p.x+12}" y="${p.y+3}">${String(n.gid).slice(-8)} · C${n.cluster_id}</text>`});graph.innerHTML=out;renderLegend()}
init();</script></body></html>'''


def serve(df, edges, out_dir, host, port, timing_ms):
    records = json.loads(df.to_json(orient="records"))
    for r in records:
        r["gid"] = str(r["gid"])  # keep 18-digit identifiers exact in JavaScript
    nodes = {r["gid"]: r for r in records}
    incident = {}
    for r in json.loads(edges.to_json(orient="records")):
        r["src"], r["dst"] = str(r["src"]), str(r["dst"])
        incident.setdefault(r["src"], []).append(r); incident.setdefault(r["dst"], []).append(r)
    top_csv = pd.read_csv(out_dir / "top_nodes.csv", dtype={"gid": "string"}).head(20)
    top_payload = []
    for r in top_csv.to_dict(orient="records"):
        node = nodes[str(r["gid"])]
        top_payload.append({"rank": int(r["rank"]), "gid": str(r["gid"]), "role": r["role"],
                            "cluster_id": node["cluster_id"], "priority_score": float(r["priority_score"])})
    kpi_payload = {"nodes": len(nodes), "edges": len(edges),
                   "clusters": int(df["cluster_id"].nunique()),
                   "turnover_kzt": float(edges["sum_kzt"].sum()), "timing_ms": int(timing_ms)}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/": body, ctype = HTML.encode(), "text/html; charset=utf-8"
            elif parsed.path == "/api/top":
                body, ctype = json.dumps(top_payload, ensure_ascii=False).encode(), "application/json"
            elif parsed.path == "/api/kpi":
                body, ctype = json.dumps(kpi_payload, ensure_ascii=False).encode(), "application/json"
            elif parsed.path == "/api/node":
                gid = parse_qs(parsed.query).get("gid", [""])[0]
                if gid not in nodes: payload = {"error": "gid not found"}
                else:
                    all_edges = incident.get(gid, [])
                    es = all_edges[:80]
                    ids = {gid} | {e["src"] for e in es} | {e["dst"] for e in es}
                    payload = {"node": nodes[gid], "nodes": [nodes[x] for x in ids], "edges": es,
                               "limited": len(all_edges) > len(es), "total_edges": len(all_edges)}
                body, ctype = json.dumps(payload, ensure_ascii=False).encode(), "application/json"
            else: self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *_): pass
    url = f"http://{host}:{port}"; print(f"UI: {url} (Ctrl+C to stop)")
    threading.Timer(.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main():
    started_at = time.perf_counter()
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=here.parent / "data", help="папка с parquet-файлами")
    ap.add_argument("--out", default=here / "out", help="куда писать выгрузки")
    ap.add_argument("--no-ui", action="store_true", help="создать CSV и выйти")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--core", type=Path, default=here.parent.parent / "cpp_core" / "build" / "bin" / "graph_core.exe")
    ap.add_argument("--python-metrics", action="store_true", help="аварийный запуск без C++")
    a = ap.parse_args()

    edges, nodes, tx = load(Path(a.data))
    sanity_check(edges, nodes, tx)
    G = build_graph(edges, nodes)
    if a.python_metrics:
        print("Metrics engine: Python fallback")
        df = basic_features(G, nodes)
    else:
        df = cpp_basic_features(G, nodes, edges, a.core)
    df = enrich(G, df, tx)
    write_outputs(df, Path(a.out), edges)
    hints(G, df)
    if not a.no_ui:
        timing_ms = round((time.perf_counter() - started_at) * 1000)
        serve(df, edges, Path(a.out), a.host, a.port, timing_ms)


if __name__ == "__main__":
    main()
