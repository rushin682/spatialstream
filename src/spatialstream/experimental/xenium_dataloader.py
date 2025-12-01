"""
PyTorch Dataset for loading tiles from Xenium SpatialData Zarr files.

This module provides a flexible Dataset class for extracting image tiles and
corresponding molecular data from SpatialData objects created from 10x Xenium data.
"""

import warnings
from collections.abc import Callable, Mapping
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
import spatialdata as sd
import torch
import torchvision.transforms.v2 as T
from geopandas import GeoDataFrame
from spatialdata._core.centroids import get_centroids
from spatialdata._core.operations.transform import transform
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    PointsModel,
    get_axes_names,
    get_model,
)
from spatialdata.transformations import (
    BaseTransformation,
    get_transformation,
    set_transformation,
)
from torch.utils.data import DataLoader, Dataset
from xarray import DataArray, DataTree

# ============================================================================
# Module-level Helper Functions
# ============================================================================


def _get_tile_coords(
    polygons: GeoDataFrame,
    cs: str,
    rasterize: bool,
    tile_scale: float | None = None,
) -> pd.DataFrame:
    """
    Get bounding box coordinates for each polygon in the target coordinate system.

    Args:
        polygons: GeoDataFrame containing polygon geometries
        cs: Target coordinate system name
        rasterize: If True, use coordinates in cs; if False, transform to intrinsic coordinates
        tile_scale: Scaling factor to extend bounding boxes (optional)

    Returns
    -------
        DataFrame with columns: [axes..., "minx", "miny", "maxx", "maxy"]
    """
    # Transform polygons to the target coordinate system
    transform(polygons, to_coordinate_system=cs)

    # If not rasterizing, we need to work in the element's intrinsic coordinate system
    if not rasterize:
        # Get the transformation to the target cs, then invert it
        transformation = get_transformation(polygons, to_coordinate_system=cs)
        assert isinstance(transformation, BaseTransformation)
        back_transformation = transformation.inverse()

        # Apply inverse transformation to get back to intrinsic coordinates
        set_transformation(polygons, back_transformation, to_coordinate_system="intrinsic_of_element")
        transform(polygons, to_coordinate_system="intrinsic_of_element")

    # Get centroids in the target coordinate system
    centroids_points = get_centroids(polygons, coordinate_system=cs)
    axes = get_axes_names(centroids_points)
    centroids_numpy = centroids_points.compute().values

    # Extract bounding box coordinates for each polygon
    bounds = polygons.geometry.bounds  # Returns DataFrame with columns: minx, miny, maxx, maxy
    min_coords = bounds[["minx", "miny"]].values
    max_coords = bounds[["maxx", "maxy"]].values

    # If tile_scale is provided, extend the bounding boxes
    if tile_scale is not None:
        # Calculate the center of each bounding box
        bbox_centers = (min_coords + max_coords) / 2

        # Calculate half-widths and half-heights
        half_extents = (max_coords - min_coords) / 2

        # Scale the extents
        scaled_half_extents = half_extents * tile_scale

        # Recalculate min and max coordinates with scaled extents
        min_coords = bbox_centers - scaled_half_extents
        max_coords = bbox_centers + scaled_half_extents

    # Construct output DataFrame with centroids and bounding box coordinates
    # Columns: ["x", "y", "minx", "miny", "maxx", "maxy"]
    return pd.DataFrame(
        np.hstack([centroids_numpy, min_coords, max_coords]),
        columns=list(axes) + ["min" + ax for ax in axes] + ["max" + ax for ax in axes],
    )


# ============================================================================
# Main Dataset Class
# ============================================================================


class XeniumDataset(Dataset):
    """
    PyTorch Dataset for extracting tiles from Xenium SpatialData objects.

    This dataset extracts image tiles and corresponding molecular data from a
    SpatialData Zarr file created from Xenium data. Currently supports spot-based
    regions only. Cell-based regions are planned for future development.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object loaded from a Zarr file.
    regions_to_images : dict[str, str]
        Mapping from region names (e.g., 'spots_55um') to image names (e.g., 'he_image').
        Region names must contain 'spots' in their name.
    regions_to_coordinate_systems : dict[str, str]
        Mapping from region names to their coordinate systems (e.g., 'global').
    target_mpp : float, default=1.0
        Target resolution in microns per pixel for output tiles.
    rasterize : bool, default=False
        Whether to rasterize shapes instead of cropping the image.
    return_annotations : str | None, default=None
        Annotation column to extract from the table. If None, uses 'in_tissue' by default.
    rasterize_kwargs : Mapping[str, Any], default={}
        Additional keyword arguments for the rasterize function.
    image_transform : Callable | None, default=None
        Torchvision transforms to apply to image tiles (e.g., normalization).
    n_cells_max : int | None, default=None
        Reserved for future cell-based region support. Currently unused.

    Examples
    --------
    >>> import spatialdata as sd
    >>> import torchvision.transforms.v2 as T
    >>> from torch.utils.data import DataLoader
    >>> # Load data
    >>> sdata = sd.read_zarr("path/to/xenium.zarr")
    >>> sdata.attrs["source_mpp"] = 0.2125
    >>> # Define transforms
    >>> transforms = T.Compose(
    ...     [
    ...         T.Resize((224, 224), antialias=True),
    ...         T.ToDtype(torch.float32, scale=True),
    ...         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ...     ]
    ... )
    >>> # Create dataset
    >>> dataset = XeniumDataset(
    ...     sdata=sdata,
    ...     regions_to_images={"spots_55um": "he_image"},
    ...     regions_to_coordinate_systems={"spots_55um": "global"},
    ...     target_mpp=0.5,
    ...     return_annotations="tissue_type",
    ...     image_transform=transforms,
    ... )
    >>> # Create dataloader
    >>> dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    """

    # Class constants
    INSTANCE_KEY = "instance_id"
    CS_KEY = "cs"
    REGION_KEY = "region"
    IMAGE_KEY = "he_image"

    def __init__(
        self,
        sdata_path: str,  # Accept path or object
        regions_to_images: dict[str, str],
        regions_to_coordinate_systems: dict[str, str],
        target_mpp: float = 1.0,
        rasterize: bool = False,
        return_annotations: str | None = None,
        rasterize_kwargs: Mapping[str, Any] = MappingProxyType({}),
        image_transform: Callable[[Any], Any] | None = None,
        n_cells_max: int | None = None,
    ):
        # --- CHANGE IS HERE: Eager Loading ---
        # Load the SpatialData object immediately and store it as an instance attribute.
        # This is the strategy from the working SpatialDataset.
        import spatialdata as sd

        self.sdata = sd.read_zarr(sdata_path)
        self.sdata_path = sdata_path  # Store path for reference if needed

        # ... rest of init ...
        self.image_transform = image_transform
        self.n_cells_max = n_cells_max  # Reserved for future cell-based regions
        self.return_annotations = return_annotations

        # Validate inputs
        self._validate(regions_to_images, regions_to_coordinate_systems)

        # Preprocess to create tile index
        self._preprocess(rasterize, target_mpp)

        # self._crop_image = partial(rasterize_fn, **dict(rasterize_kwargs)) if rasterize else bounding_box_query
        self.rasterize = rasterize
        self.rasterize_kwargs = dict(rasterize_kwargs)

    # ------------------------------------------------------------------------
    # Public Methods (PyTorch Dataset Interface)
    # ------------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of tiles in the dataset."""
        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """
        Get a single tile by index.

        Parameters
        ----------
        idx : int
            Index of the tile to retrieve.

        Returns
        -------
        dict[str, Any] | None
            Dictionary with:
                - 'image': torch.Tensor, transformed image tile
                - 'spot_gexp': torch.Tensor, 1D gene expression vector
                - 'spot_coords': torch.Tensor, (x, y) spatial coordinates
                - 'annotation': annotation value for this spot
                - 'image_coords': np.ndarray (3, 2), [[minx, miny], [maxx, maxy], [centroidx, centroidy]]
            Returns None if the tile extraction fails.
        """
        # Ensure sdata is loaded in this worker

        # Get tile metadata
        row = self.dataset_index.iloc[idx]
        t_coords = self.tiles_coords.iloc[idx]
        # cs = row[self.CS_KEY] # caused linting errors. Will uncomment later (26th Nov 2025)
        region_name = row[self.REGION_KEY]
        instance_id = str(row[self.INSTANCE_KEY])

        # Get table and extract spot data
        table = self._get_table(region_name)

        # Index into table using instance_id
        try:
            spot_idx = table.obs.index.get_loc(instance_id)
        except KeyError:
            warnings.warn(
                f"Instance ID {instance_id} not found in table for region {region_name}",
                UserWarning,
                stacklevel=2,
            )
            return None

        # Extract gene expression data
        if scipy.sparse.issparse(table.X):
            expression = table.X[spot_idx].toarray().flatten()
        else:
            expression = table.X[spot_idx]

        # Extract spatial coordinates
        spatial = table.obsm["spatial"][spot_idx]

        # Get annotation value
        if self.return_annotations:
            annotation_col = self.return_annotations
        else:
            annotation_col = "in_tissue"

        try:
            annotation_value = table.obs.loc[instance_id, annotation_col]
        except KeyError:
            warnings.warn(
                f"Annotation column '{annotation_col}' not found in table.obs for region {region_name}",
                UserWarning,
                stacklevel=2,
            )
            annotation_value = None

        # Prepare spot data dictionary
        spot_data = {
            "expression": expression,
            "spatial": spatial,
            "instance_id": instance_id,
            "annotation_value": annotation_value,
        }

        # Crop the image tile
        image_element = self.sdata[row[self.IMAGE_KEY]]
        # if self.rasterize:
        #     tile = rasterize_fn(
        #         image_element,
        #         axes=tuple(self.dims),
        #         min_coordinate=t_coords[[f"min{i}" for i in self.dims]].values,
        #         max_coordinate=t_coords[[f"max{i}" for i in self.dims]].values,
        #         target_coordinate_system=cs,
        #         **self.rasterize_kwargs
        #     )
        # else:
        #     tile = bounding_box_query(
        #         image_element,
        #         axes=tuple(self.dims),
        #         min_coordinate=t_coords[[f"min{i}" for i in self.dims]].values,
        #         max_coordinate=t_coords[[f"max{i}" for i in self.dims]].values,
        #         target_coordinate_system=cs,
        #     )

        patch = image_element.isel(
            x=slice(int(t_coords["minx"]), int(t_coords["maxx"])), y=slice(int(t_coords["miny"]), int(t_coords["maxy"]))
        )
        # patch = patch.transpose(1, 2, 0)

        # Process and return the tile
        return self._process_tile(patch, t_coords, spot_data, region_name)

    # ------------------------------------------------------------------------
    # Validation Methods
    # ------------------------------------------------------------------------

    def _validate(
        self,
        regions_to_images: dict[str, str],
        regions_to_coordinate_systems: dict[str, str],
    ) -> None:
        """
        Validate that the regions, images, and coordinate systems are compatible.

        Parameters
        ----------
        regions_to_images : dict[str, str]
            Mapping from region to image names.
        regions_to_coordinate_systems : dict[str, str]
            Mapping from region to coordinate system names.

        Raises
        ------
        ValueError
            If regions, images, or coordinate systems are incompatible.
        NotImplementedError
            If any region does not contain 'spots' in its name.
        """
        # Check that both dicts have the same keys
        if set(regions_to_images.keys()) != set(regions_to_coordinate_systems.keys()):
            raise ValueError("Keys in regions_to_images and regions_to_coordinate_systems must match.")

        self.regions = list(regions_to_coordinate_systems.keys())

        # Check that all regions are spot-based
        for region_name in self.regions:
            if "spots" not in region_name:
                raise NotImplementedError(
                    f"Cell-based regions are not yet implemented. "
                    f"Only spot-based regions (with 'spots' in the name) are currently supported. "
                    f"Got region: '{region_name}'"
                )

        cs_region_image: list[tuple[str, str, str]] = []

        # Validate each region-image-coordinate system triplet
        for region_name in self.regions:
            image_name = regions_to_images[region_name]
            region_elem = self.sdata[region_name]
            image_elem = self.sdata[image_name]

            # Check element types
            if get_model(region_elem) == PointsModel:
                raise ValueError("Region must be a shapes or labels element, not points.")
            if get_model(image_elem) not in [Image2DModel, Image3DModel]:
                raise ValueError("Image must be a 2D or 3D image element.")

            # Check coordinate system compatibility
            cs = regions_to_coordinate_systems[region_name]
            region_trans = get_transformation(region_elem, get_all=True)
            image_trans = get_transformation(image_elem, get_all=True)

            if not (
                isinstance(region_trans, dict)
                and isinstance(image_trans, dict)
                and cs in region_trans
                and cs in image_trans
            ):
                raise ValueError(
                    f"Coordinate system '{cs}' not found for region '{region_name}' and image '{image_name}'."
                )

            cs_region_image.append((cs, region_name, image_name))

        self._cs_region_image = tuple(cs_region_image)

    # ------------------------------------------------------------------------
    # Preprocessing Methods
    # ------------------------------------------------------------------------

    def _preprocess(self, rasterize: bool, target_mpp: float) -> None:
        """
        Preprocess the dataset by computing tile coordinates and creating an index.

        Parameters
        ----------
        rasterize : bool
            Whether tiles will be rasterized.
        target_mpp : float
            Target resolution for tiles.
        """
        index_df_list = []
        tile_coords_list = []
        dims_list = []

        # Process each region
        for cs, region_name, image_name in self._cs_region_image:
            patch_polygons = self.sdata[region_name]
            dims_list.append(get_axes_names(patch_polygons))

            # Compute tile coordinates
            tile_coords = _get_tile_coords(
                polygons=patch_polygons,
                cs=cs,
                rasterize=rasterize,
                tile_scale=target_mpp / self.sdata.attrs["source_mpp"],
            )
            tile_coords_list.append(tile_coords)

            # Create index dataframe
            inst = patch_polygons.index.values
            df = pd.DataFrame(
                {
                    self.INSTANCE_KEY: inst,
                    self.CS_KEY: cs,
                    self.REGION_KEY: region_name,
                    self.IMAGE_KEY: image_name,
                }
            )
            index_df_list.append(df)

        # Store preprocessed data
        self.patch_polygons = patch_polygons
        self.tiles_coords = pd.concat(tile_coords_list, ignore_index=True)
        self.dataset_index = pd.concat(index_df_list, ignore_index=True)

        # Validate preprocessing
        if len(self.tiles_coords) != len(self.dataset_index):
            raise RuntimeError("Mismatch between tile coordinates and dataset index.")

        # Store dimension information
        dims_set = set(chain(*dims_list))
        if not all(dim in self.tiles_coords.columns for dim in dims_set):
            raise ValueError("Some dimensions are missing in tile coordinates.")
        self.dims = list(dims_set)

    # ------------------------------------------------------------------------
    # Data Processing Methods
    # ------------------------------------------------------------------------

    def _process_tile(
        self,
        image: DataArray | DataTree,
        t_coords: pd.Series,
        spot_data: dict[str, Any],
        region_name: str,
    ) -> dict[str, Any]:
        """
        Process a tile and return the formatted output.

        Parameters
        ----------
        image : DataArray | DataTree
            The cropped image tile.
        t_coords : pd.Series
            Tile coordinate information with columns: ["x", "y", "minx", "miny", "maxx", "maxy"].
        spot_data : dict[str, Any]
            Dictionary containing:
                - 'expression': 1D numpy array of gene expression
                - 'spatial': (x, y) tuple of spatial coordinates
                - 'instance_id': spot identifier
                - 'annotation_value': annotation value for this spot
        region_name : str
            Name of the region this tile belongs to.

        Returns
        -------
        dict[str, Any]
            Processed tile data with keys:
                - 'image': transformed image tensor
                - 'spot_gexp': 1D gene expression tensor
                - 'spot_coords': (x, y) coordinate tensor
                - 'annotation': annotation value
                - 'image_coords': (3, 2) array of bounding box and centroid coordinates

        Raises
        ------
        NotImplementedError
            If the region is not spot-based (doesn't contain 'spots' in name).
        """
        # Check if this is a spot-based region
        if "spots" not in region_name:
            raise NotImplementedError(
                f"Cell-based region processing not yet implemented for region '{region_name}'. "
                "This feature is planned for future development."
            )

        # Process image
        image = self._ensure_single_scale(image)
        image_np = image.data.compute()
        # image_np = image_np.transpose(1,2,0)
        image_tensor = torch.from_numpy(image_np)

        # Apply transforms if provided
        if self.image_transform:
            image_tensor = self.image_transform(image_tensor)

        # Extract spot data as tensors
        spot_gexp = torch.tensor(spot_data["expression"], dtype=torch.float32)  # 1D tensor
        spot_coords = torch.tensor(spot_data["spatial"], dtype=torch.float32)  # (2,) tensor

        # Get annotation value
        annotation = torch.tensor(self._extract_annotation(spot_data["annotation_value"]), dtype=torch.float32)

        # Build image_coords from t_coords: [[minx, miny], [maxx, maxy], [centroid_x, centroid_y]]
        image_coords = torch.tensor(
            [
                [t_coords["minx"], t_coords["miny"]],
                [t_coords["maxx"], t_coords["maxy"]],
                [t_coords["x"], t_coords["y"]],  # centroid
            ],
            dtype=torch.float32,
        )

        return {
            "image": image_tensor,
            "spot_gexp": spot_gexp,
            "spot_coords": spot_coords,
            "annotation": annotation,
            "image_coords": image_coords,
        }

    def _get_table(self, region_name: str):
        """
        Get the appropriate table based on the region name.

        For regions starting with 'spots_', uses the corresponding spot table.
        For other regions, uses the main 'table'.

        Parameters
        ----------
        region_name : str
            Name of the region.

        Returns
        -------
        AnnData
            The appropriate table for this region.

        Raises
        ------
        KeyError
            If the required table is not found.
        """
        table_name = f"{region_name}_table" if region_name.startswith("spots_") else "table"

        if table_name not in self.sdata.tables:
            raise KeyError(
                f"Required table '{table_name}' not found in SpatialData object. "
                f"Available tables: {list(self.sdata.tables.keys())}"
            )

        return self.sdata.tables[table_name]

    def _extract_annotation(self, annotation_value: Any) -> Any:
        """
        Extract the annotation value for a spot.

        Parameters
        ----------
        annotation_value : Any
            The annotation value to return.

        Returns
        -------
        Any
            The annotation value (pass-through for now, with potential for future processing).
        """
        return annotation_value

    # ------------------------------------------------------------------------
    # Methods Reserved for Future Cell-Based Region Support
    # ------------------------------------------------------------------------
    def _apply_padding(
        self, expression_tensor: torch.Tensor, spatial_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad or truncate tensors to match n_cells_max.

        RESERVED FOR FUTURE USE: This method will be used when cell-based regions are implemented.

        Parameters
        ----------
        expression_tensor : torch.Tensor
            Gene expression matrix (n_cells x n_genes).
        spatial_tensor : torch.Tensor
            Spatial coordinates (n_cells x 2).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Padded expression tensor, padded spatial tensor, and padding mask.
        """
        if self.n_cells_max is None:
            raise ValueError("n_cells_max must be set to use padding functionality.")

        n_cells = expression_tensor.shape[0]
        padding = self.n_cells_max - n_cells

        if padding > 0:
            # Add padding
            mask = torch.cat([torch.ones(n_cells), torch.zeros(padding)])
            expr_pad = torch.zeros(padding, expression_tensor.shape[1])
            spatial_pad = torch.full((padding, spatial_tensor.shape[1]), -1.0)

            expression_tensor = torch.cat([expression_tensor, expr_pad], dim=0)
            spatial_tensor = torch.cat([spatial_tensor, spatial_pad], dim=0)
        else:
            # Truncate if necessary
            mask = torch.ones(self.n_cells_max)
            expression_tensor = expression_tensor[: self.n_cells_max]
            spatial_tensor = spatial_tensor[: self.n_cells_max]

        return expression_tensor, spatial_tensor, mask

    # ------------------------------------------------------------------------
    # Static Utility Methods
    # ------------------------------------------------------------------------

    @staticmethod
    def _ensure_single_scale(data: DataArray | DataTree) -> DataArray:
        """
        Extract a single-scale DataArray from multi-scale data.

        Parameters
        ----------
        data : DataArray | DataTree
            The input data, possibly multi-scale.

        Returns
        -------
        DataArray
            A single-scale DataArray (scale0 if multi-scale).

        Raises
        ------
        TypeError
            If data is neither DataArray nor DataTree.
        """
        if isinstance(data, DataArray):
            return data
        if isinstance(data, DataTree):
            return next(iter(data["scale0"].ds.values()))
        raise TypeError(f"Expected DataArray or DataTree, got {type(data)}.")


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("XeniumDataset Usage Example")
    print("=" * 70)
    print("\nNOTE: This example requires actual Zarr files to run.\n")

    # Example transformation pipeline
    image_transforms = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Example 1: Load from pseudo-spots")
    print("-" * 70)

    zarr_path = Path("/p/project1/hai_spatial_clip/spatialrefinery/example_data")
    zarr_file = "Xenium_V1_hKidney_cancer_section.zarr"
    sdata = sd.read_zarr(zarr_path / zarr_file)

    dataset = XeniumDataset(
        sdata_path=str(zarr_path / zarr_file),
        regions_to_images={"spots_55um": "he_image"},
        regions_to_coordinate_systems={"spots_55um": "global"},
        target_mpp=0.5,
        return_annotations="in_tissue",
        image_transform=image_transforms,
        rasterize=True,
        rasterize_kwargs={"target_unit_to_pixels": 1.0},
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    # Iterate through batches
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break
        print(f"Batch {i + 1}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Gene expression shape: {batch['spot_gexp'].shape}")
        print(f"  Annotations: {batch['annotation']}")
