import os
import json
import re
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import openai
except ImportError:
    openai = None

from src.data.dataset import CANDIDATE_LOCATIONS


SYSTEM_PROMPT = """You are an emergency incident prediction expert specializing in Australian bushfire analysis (2019-2020 Black Summer).

Your task is to predict fire evolution for the next 1-hour window based on multi-source data (satellite + social media).

IMPORTANT CONSTRAINTS:
- For affected_areas, you MUST choose ONLY from this list: {locations}
- risk_score must be between 0 and 1
- predicted_direction is in degrees (0=North, 90=East, 180=South, 270=West)

Always respond in this exact JSON format:
{{
    "risk_score": <float 0-1>,
    "risk_level": "<low|medium|high>",
    "predicted_direction": <degrees 0-360>,
    "affected_areas": ["<from approved list>"],
    "explanation": "<reasoning>"
}}""".format(locations=", ".join(CANDIDATE_LOCATIONS))


@dataclass
class LLMPrediction:
    risk_score: float
    risk_level: str
    predicted_direction: float
    affected_areas: List[str]
    explanation: str
    raw_response: str = ""


def build_prompt(graph_stats: dict, top_edges: list, social_texts: list,
                 dynamics: dict, timestamp: str) -> str:
    lines = [
        f"Timestamp: {timestamp}",
        f"Region: NSW South Coast / East Gippsland, Australia",
        "",
        "=== Graph Topology ===",
        f"Physical sensor nodes: {graph_stats.get('num_physical_nodes', 0)}",
        f"Social media nodes: {graph_stats.get('num_social_nodes', 0)}",
        f"Edges: {graph_stats.get('num_edges', 0)} (density: {graph_stats.get('density', 0):.3f})",
        f"GCC ratio: {graph_stats.get('gcc_ratio', 0):.3f}",
        f"Avg edges/physical: {graph_stats.get('avg_edges_per_physical', 0):.1f}",
        f"Sparsemax temperature: {graph_stats.get('sparsemax_temperature', 1.0):.3f}",
        "",
        "=== Top Connections ===",
    ]
    for e in top_edges[:5]:
        lines.append(f"  social_{e.source_id} -> physical_{e.target_id} "
                     f"(weight={e.weight:.3f}, unc={e.data_uncertainty:.3f})")
    lines.append("")
    lines.append("=== Social Media Evidence ===")
    for i, text in enumerate(social_texts[:7], 1):
        lines.append(f"  {i}. \"{text[:120]}\"")
    lines.append("")
    lines.append("=== Fire Dynamics ===")
    lines.append(f"  Current centroid: ({dynamics.get('centroid_lat', 0):.4f}, {dynamics.get('centroid_lon', 0):.4f})")
    lines.append(f"  Speed: {dynamics.get('speed_kmh', 0):.1f} km/h")
    lines.append(f"  Direction: {dynamics.get('direction', 0):.0f}")
    lines.append(f"  Growth rate: {dynamics.get('growth_rate', 0)*100:+.1f}%")
    lines.append(f"  Fire points (current window): {dynamics.get('fire_count_curr', 0)}")
    lines.append("")
    lines.append(f"Candidate locations for affected_areas: {', '.join(CANDIDATE_LOCATIONS)}")
    lines.append("")
    lines.append("Predict the fire evolution for the next 1-hour window. Return JSON only.")
    return "\n".join(lines)


def call_gpt4o(prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
    if openai is None:
        raise ImportError("openai not installed")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def parse_llm_response(response: str, fallback_direction: float = 0.0) -> LLMPrediction:
    try:
        match = re.search(r"\{[\s\S]*\}", response)
        if match:
            data = json.loads(match.group())
            areas = [a for a in data.get("affected_areas", [])
                     if a in CANDIDATE_LOCATIONS]
            return LLMPrediction(
                risk_score=float(data.get("risk_score", 0.5)),
                risk_level=data.get("risk_level", "medium"),
                predicted_direction=float(data.get("predicted_direction", fallback_direction)),
                affected_areas=areas,
                explanation=data.get("explanation", ""),
                raw_response=response,
            )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return LLMPrediction(
        risk_score=0.5,
        risk_level="medium",
        predicted_direction=fallback_direction,
        affected_areas=[],
        explanation=f"Parse failed: {response[:200]}",
        raw_response=response,
    )


def run_llm_reasoning(
    graph_stats: dict,
    edges: list,
    social_texts: list,
    dynamics: dict,
    timestamp: str,
) -> LLMPrediction:
    top_edges = sorted(edges, key=lambda e: e.weight, reverse=True)[:5]
    prompt = build_prompt(graph_stats, top_edges, social_texts, dynamics, timestamp)
    try:
        response = call_gpt4o(prompt)
        prediction = parse_llm_response(response, dynamics.get("direction", 0))
    except Exception as e:
        prediction = LLMPrediction(
            risk_score=0.5, risk_level="medium",
            predicted_direction=dynamics.get("direction", 0),
            affected_areas=[], explanation=f"API error: {e}",
        )
    return prediction
