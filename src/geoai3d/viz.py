"""Interactive 3D viewing of point clouds in Jupyter and Colab.

:func:`view` renders a cloud as an interactive plotly scatter that displays
inline in a notebook, coloured by height, an attribute (class, intensity, a
geometric feature, ...), or true RGB colour when present. Large clouds are
randomly thinned for display so the browser stays responsive.

plotly is an optional dependency: install it with ``pip install geoai3d[viz]``.
It is pure Python, so it needs no compiler and works the same in Colab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from geoai3d.subsample import subsample

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from geoai3d.core.pointcloud import PointCloud

_DEFAULT_MAX_POINTS = 100_000


def _to_8bit(values: NDArray[Any]) -> NDArray[np.int64]:
    """Scale colour values to 0-255, handling 16-bit LAS colour."""
    array = np.asarray(values)
    if array.size and array.max() > 255:
        array = array // 256
    return array.astype(np.int64)


def _resolve_color(
    cloud: PointCloud, color_by: str | None
) -> tuple[Any, str | None, str | None]:
    """Return (colour values, colorscale, colorbar title) for the display.

    A ``None`` colorscale signals per-point RGB colour strings.
    """
    names = cloud.attribute_names
    has_rgb = {"red", "green", "blue"} <= set(names)
    if color_by == "rgb" or (color_by is None and has_rgb):
        red = _to_8bit(cloud.attribute("red"))
        green = _to_8bit(cloud.attribute("green"))
        blue = _to_8bit(cloud.attribute("blue"))
        colors = [f"rgb({r},{g},{b})" for r, g, b in zip(red, green, blue, strict=True)]
        return colors, None, None
    if color_by is None or color_by in ("z", "height"):
        return cloud.xyz[:, 2], "Viridis", "height"
    if color_by in names:
        return cloud.attribute(color_by), "Viridis", color_by
    msg = (
        f"Cannot colour by {color_by!r}. Choose an attribute from {names}, "
        "or 'z'/'height', or 'rgb'."
    )
    raise ValueError(msg)


def view(
    cloud: PointCloud,
    *,
    color_by: str | None = None,
    max_points: int = _DEFAULT_MAX_POINTS,
    point_size: float = 2.0,
    seed: int = 0,
) -> go.Figure:
    """Render a point cloud as an interactive 3D plot.

    Args:
        cloud: The cloud to display.
        color_by: Attribute name to colour by, or ``"z"``/``"height"``, or
            ``"rgb"`` for true colour. Defaults to RGB when red/green/blue
            attributes are present, otherwise height.
        max_points: Randomly thin the cloud to at most this many points for
            display, so the browser stays responsive.
        point_size: Marker size.
        seed: Seed for the display thinning, for a reproducible view.

    Returns:
        A ``plotly.graph_objects.Figure``. In a notebook it renders inline;
        elsewhere call ``.show()`` on it.

    Raises:
        ImportError: If plotly is not installed.
        ValueError: If ``color_by`` names something that cannot be used.

    Example:
        >>> import numpy as np
        >>> from geoai3d import PointCloud
        >>> from geoai3d.viz import view
        >>> cloud = PointCloud(np.random.default_rng(0).random((500, 3)), crs=28992)
        >>> figure = view(cloud)
        >>> figure.data[0].type
        'scatter3d'
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        msg = (
            "view() needs plotly. Install the viewer extra with "
            "'pip install geoai3d[viz]'."
        )
        raise ImportError(msg) from exc

    display = cloud
    if len(cloud) > max_points:
        display = subsample(cloud, method="random", count=max_points, seed=seed)

    xyz = display.xyz
    color, colorscale, colorbar_title = _resolve_color(display, color_by)
    if colorscale is None:
        marker = {"size": point_size, "color": color}
    else:
        marker = {
            "size": point_size,
            "color": color,
            "colorscale": colorscale,
            "colorbar": {"title": colorbar_title},
        }

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=marker,
            )
        ]
    )
    figure.update_layout(
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    return figure
