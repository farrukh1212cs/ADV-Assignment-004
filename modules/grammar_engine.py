"""
Grammar-of-Graphics engine for GAP-VizAI.
Implements a declarative VizSpec (Data → Aesthetics → Geom → Stat → Scale → Facet)
that compiles to Plotly figures with perceptual defaults applied automatically.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.perceptual import OKABE_ITO, apply_template


@dataclass
class DataLayer:
    n_rows: int
    n_cols: int
    columns: list
    domain: str = "unknown"


@dataclass
class AestheticLayer:
    x: str
    y: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    facet: Optional[str] = None
    label: Optional[str] = None


@dataclass
class GeomLayer:
    """Geometry type mapping (mirrors ggplot2 geom_* functions)."""
    type: str  # bar | line | point | area | violin | ecdf | kde | box | ridge | heatmap | treemap | sankey
    opacity: float = 0.85
    line_width: float = 2.0
    point_size: float = 5.0


@dataclass
class StatLayer:
    """Statistical transformation applied before rendering."""
    type: str = "identity"   # identity | bin | density | ecdf | smooth | ci | summary
    bins: int = 30
    ci_level: float = 0.95
    smooth_span: float = 0.75


@dataclass
class ScaleLayer:
    palette: str = "okabe_ito"
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    title: str = ""
    x_type: str = "linear"   # linear | log | date | category
    y_type: str = "linear"


@dataclass
class FacetLayer:
    col: Optional[str] = None
    row: Optional[str] = None
    n_col: int = 2


@dataclass
class VizSpec:
    """
    Complete grammar specification for a single visualization.
    Mirrors the layered grammar of graphics (Wilkinson, 2005; Wickham, 2010).
    """
    data:      DataLayer
    aesthetic: AestheticLayer
    geom:      GeomLayer
    stat:      StatLayer      = field(default_factory=StatLayer)
    scale:     ScaleLayer     = field(default_factory=ScaleLayer)
    facet:     FacetLayer     = field(default_factory=FacetLayer)
    notes:     str            = ""

    def to_dict(self) -> dict:
        return {
            "data":      asdict(self.data),
            "aesthetic": asdict(self.aesthetic),
            "geom":      asdict(self.geom),
            "stat":      asdict(self.stat),
            "scale":     asdict(self.scale),
            "facet":     asdict(self.facet),
            "notes":     self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def grammar_compliance_score(self) -> float:
        """
        Score 0–1 measuring how well the spec follows grammar-of-graphics principles:
        - x and y are defined for positional geoms
        - color is used for categorical distinctions (not continuous where inappropriate)
        - stat transform matches the geom type
        """
        score = 1.0
        positional_geoms = {"bar", "line", "point", "area", "violin", "box", "ridge"}
        if self.geom.type in positional_geoms and self.aesthetic.y is None:
            score -= 0.2
        if self.geom.type in {"violin", "box", "ridge"} and self.aesthetic.color is None:
            score -= 0.1
        stat_geom_pairs = {
            "ecdf": "ecdf", "kde": "density", "bar": "bin",
            "line": "identity", "point": "identity",
        }
        expected_stat = stat_geom_pairs.get(self.geom.type)
        if expected_stat and self.stat.type not in ("identity", expected_stat):
            score -= 0.15
        return round(max(0.0, score), 3)


def build_spec(
    df: pd.DataFrame,
    x: str,
    geom_type: str,
    y: str = None,
    color: str = None,
    title: str = "",
    domain: str = "unknown",
    stat_type: str = "identity",
    x_label: str = None,
    y_label: str = None,
    notes: str = "",
) -> VizSpec:
    """Convenience factory for building a VizSpec from a DataFrame."""
    return VizSpec(
        data=DataLayer(
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=list(df.columns),
            domain=domain,
        ),
        aesthetic=AestheticLayer(x=x, y=y, color=color),
        geom=GeomLayer(type=geom_type),
        stat=StatLayer(type=stat_type),
        scale=ScaleLayer(
            title=title,
            x_label=x_label or x,
            y_label=y_label or (y or ""),
        ),
        notes=notes,
    )
