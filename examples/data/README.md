# Tutorial sample data

`ahn_sample.laz` is a small (~150 m) window cropped from the Dutch national
LiDAR dataset **AHN4**, tile `45BZ2_13`, obtained via the GeoTiles catalogue
(<https://geotiles.citg.tudelft.nl/>).

The raw AHN point cloud is released by the Dutch government into the **public
domain (CC0)**, so this crop is redistributed here freely, with no attribution
required. It is included only so the tutorial notebooks run out of the box.

The window was chosen over a built-up area so the tutorials have real structure
to work with. It carries AHN's standard ASPRS-style classification:

| class | meaning        | share |
|-------|----------------|-------|
| 2     | ground         | ~70%  |
| 6     | building       | ~23%  |
| 1     | unclassified / vegetation | ~7% |
| 9     | water          | ~0.1% |

Coordinate reference system: **EPSG:7415** — a compound of RD New (EPSG:28992)
horizontal and NAP (EPSG:5709) vertical.

To regenerate or replace it, crop a built-up window from any AHN tile with
`geoai3d.read_lidar` + `geoai3d.to_las` (see the project discussion for the
crop script).
