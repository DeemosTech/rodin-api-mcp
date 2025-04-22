import mimetypes
from pathlib import Path
import pathlib
from typing import Any, Optional, Literal, Annotated, Self, Type, Union, IO
import json
from jsonschema import validate, ValidationError
import os
from pydantic import BaseModel, Field, PositiveInt, WithJsonSchema, field_validator, model_validator, ConfigDict

def with_docstring_description(model_cls: Type[BaseModel]) -> Type[BaseModel]:
    doc = (model_cls.__doc__ or "").strip()
    old_config = getattr(model_cls, 'model_config', ConfigDict())
    merged_config = ConfigDict(**old_config, json_schema_extra={"description": doc})
    model_cls.model_config = merged_config
    return model_cls

type_httpx_file_with_mime = tuple[
    Annotated[Optional[str], 'File name'],
    Annotated[Union[str, bytes, IO], "File Content"],
    Annotated[Optional[str], "MIME"]
]

type_httpx_file_without_mime = tuple[
    Annotated[Optional[str], 'File name'],
    Annotated[Union[str, bytes, IO], "File Content"]
]

type_httpx_file_field_tuple = tuple[
    Annotated[str, "Field name"],
    type_httpx_file_with_mime | type_httpx_file_without_mime
]

with open(
    os.path.join(
        os.path.dirname(__file__), 
        "RodinParametersConstraints.json"
    )
) as f:
    rodin_parameters_constraints = json.load(f)

class RodinParameters(BaseModel):
    """
    Parameter for generate_3d_model.

    Avoid giving default value to arguments already having default value.
    """
    image_paths: list[str] = Field(
        default_factory=list,
        min_length=0,
        max_length=5,
        description="""
        Absolute file path to images to be used in generation. The first image will also be the image for material generation.
        For Image-to-3D generation: required (one or more images are needed, maximum 5 images)
        For Text-to-3D generation: Empty list
        """
    )
    prompt: Optional[str] = Field(
        default=None,
        min_length=1,
        description="""
        A textual prompt to guide the model generation.
        For Image-to-3D generation: optional (if not provided, an AI-generated prompt based on the provided images will be used)
        For Text-to-3D generation: required
        """
    )
    condition_mode: Literal['concat', 'fuse'] = Field(
        default='concat',
        description="""
        Useful only for multi-images 3D generation.
        This is an optional parameter that chooses the mode of the multi-image geneartion. Possible values are fuse and concat. Default is concat.
        For fuse mode, if you are uploading images of multiple objects, fuse mode will extract and fuse all the features of all the objects from the images for generation. One or more images are required.
        For concat mode, if you are uploading images of a single object, concat mode will inform the Rodin model to expect these images to be multi-view images of a single object. One or more images are required (you can upload multi-view images in any order, regardless of the order of view.)
        """
    )
    seed: Optional[int] = Field(
        default=None,
        ge=0,
        le=65535,
        description="""
        A seed value for randomization in the mesh generation. If not provided, the seed will be randomly generated.
        """
    )
    geometry_file_format: Literal['glb', 'usdz', 'fbx', 'obj', 'stl'] = Field(
        default='glb',
        description="""
        The format of the output geometry file.
        For Rodin Sketch, the value will fixed to glb.
        Always confirm the geometry type with the user before generation.
        """
    )
    material: Literal['PBR', 'Shaded'] = Field(
        default='PBR',
        description="""
        The material type.
        For Rodin Sketch, the value will fixed to PBR.
        """
    )
    quality: Literal['high', 'medium', 'low', 'extra-low'] = Field(
        default='medium',
        description="""
        The generation quality. 
        Possible values are high(50k faces), medium(18k faces), low(8k faces), and extra-low(4k faces).
        For Rodin Sketch or when Raw mode, the value will fixed to medium.
        """
    )
    use_hyper: bool = Field(
        default=False,
        description="""
        In generating objects with finer structure details, the quality of the 3D representation would be better if the user_hyper parameter is set to true.
        For Rodin Sketch, the value is always false.
        """
    )
    tier: Literal['Regular', 'Sketch'] = Field(
        default='Regular',
        description="Tier of generation."
    )
    TAPose: bool = Field(
        default=False,
        description="""
        When generating the human-like model, this parameter control the generation result to T/A Pose.
        When true, your model will be either T pose or A pose.
        """
    )
    bbox_condition: Optional[
        Annotated[
            list[PositiveInt], # Apply PositiveInt constraint to list items
            Field(min_length=3, max_length=3) # Apply length constraint to the list itself
        ]
    ] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="""
        This is a controlnet that controls the maxmimum sized of the generated model.
        This array must contain 3 elements, Width(Y-axis), Height(Z-axis), and Length(X-axis), in this exact fixed sequence (y, z, x).
        """
    )
    mesh_mode: Literal['Quad', 'Raw'] = Field(
        default='Quad',
        description="""
        It controls the type of faces of generated models, Possible values are Raw and Quad. Default is Quad.
        The Raw mode generates triangular face models.
        The Quad mode generates quadrilateral face models.
        When its value is Raw, quality will be fixed to medium, and addons will be fixed to [].
        """
    )
    mesh_simplify: Optional[bool] = Field(
        default=None,
        description="""
        If true, The generated models will be simplified.
        This parameter takes effect when the mesh_mode is set to Raw.
        """
    )
    mesh_smooth: Optional[bool] = Field(
        default=None,
        description="""
        If true, The generated models will be smoothed. Similar to Rodin Gen-1.
        This parameter takes effect when the mesh_mode is set to Quad.
        """
    )
    addons: list[Literal['HighPack']] = Field(
        default_factory=list,
        description="""
        By selecting HighPack, it will generate 4K resolution texture instead of the default 2K, and the number of faces will be ~16 times of the number of faces selected in the quality parameter.
        """
    )

    model_config=ConfigDict(
        json_schema_extra=rodin_parameters_constraints
    )

    @field_validator('image_paths', mode='after')
    @classmethod
    def check_image_paths(cls, value: list) -> list:
        for image_path in value:
            if not os.path.isabs(image_path):
                raise ValueError(f"'{image_path}' is not an absolute path")
            if not os.access(image_path, os.R_OK):
                raise ValueError(f"MCP Server do not have access to the image file {image_path}")
            if not os.path.isfile(image_path):
                raise ValueError(f"'{image_path}' is not a valid file")
            guessed_mime = mimetypes.guess_file_type(image_path)[0]
            if not guessed_mime.startswith('image/'):
                raise ValueError(f"'{image_path}' does not seem like an image file")
        return value
    

    @model_validator(mode='after')
    def validate_schema(self) -> Self:
        try:
            # Use the JSON schema in model_config to validate the instance data
            validate(instance=self.model_dump(), schema=RodinParameters.model_json_schema())
        except ValidationError as e:
            # If validation fails, raise an error with the validation message
            raise ValueError({
                "message": e.message,
                "context": e.context,
                "cause": e.cause,
                "instance": e.instance,
                "json_path": e.json_path,
                "schema": e.schema,
            })
        return self

    def convert_to_files(self) -> list[
        tuple[
            Annotated[str, "Field name"],
            tuple[
                Annotated[Optional[str], 'File name'],
                Annotated[Union[str, bytes, IO], "File Content"],
                Annotated[Optional[str], "MIME"]
            ]
        ]
    ]:
        result = [
            ("condition_mode", (None, self.condition_mode)),
            ("geometry_file_format", (None, self.geometry_file_format)),
            ("material", (None, self.material)),
            ("quality", (None, self.quality)),
            ("use_hyper", (None, json.dumps(self.use_hyper))),
            ("tier", (None, self.tier)),
            ("TAPose", (None, json.dumps(self.TAPose))),
            ("mesh_mode", (None, self.mesh_mode)),
        ]
        if self.seed is not None:
            result.append(("seed", (None, str(self.seed))))
        if self.mesh_simplify is not None:
            result.append(("mesh_simplify", (None, json.dumps(self.mesh_simplify))))
        if self.mesh_smooth is not None:
            result.append(("mesh_simplify", (None, json.dumps(self.mesh_smooth))))
        if self.bbox_condition:
            result.append(("bbox_condition", (None, json.dumps(self.bbox_condition))))
        if self.prompt:
            result.append(("prompt", (None, self.prompt)))
        for addon in self.addons:
            result.append(("addons", (None, addon)))

        for i, image_path in enumerate(self.image_paths):
            path_suffix = Path(image_path).suffix
            with open(image_path, "rb") as f:
                result.append(
                    (
                        "images", (
                            f"{i:04d}{path_suffix}",
                            f.read(),
                            mimetypes.guess_file_type(image_path)[0]
                        )
                    )
                )
        
        return result

@with_docstring_description
class DownloadRequestParameters(BaseModel):
    """
    Parameter for try_download_result.

    Avoid giving default value to arguments already having default value.
    """
    subscription_key: str = Field(description="The subscription_key returned by generated_3d_model")
    uuid: str = Field(description="The uuid returned by generated_3d_model")
    download_to_path: str = Field(description="Where the asset will be stored. Must ask for tha path from user before the tool call. Guessing a path without user giving the path is FORBIDDEN. Must be an absolute path.")
    # poll_interval: float = Field(default=2.5, gt=0, description="Interval between generation task status poll")
    # retry_count: int = Field(default=5, gt=0, description="Max retry count for generation task status poll")

    @field_validator('download_to_path')
    @classmethod
    def validate_resolve_and_make_absolute(cls, v: str) -> str:
        """
        Validates input path is not empty, resolves it using expanduser() and resolve(),
        and returns the absolute path string. Does NOT check for existence here.
        """
        if not v:
            raise ValueError("Input download path cannot be empty.")
        try:
            resolved_path = pathlib.Path(v).expanduser()
            if not resolved_path.is_absolute():
                 # This case is highly unlikely if resolve() succeeds but good robustness
                 raise ValueError(f"Path resolution failed to produce an absolute path for '{v}'.")
            return str(resolved_path)
        except RuntimeError as e: # Catches potential issues like maximum symlink depth exceeded
            raise ValueError(f"Error resolving path '{v}': {e}") from e
        except OSError as e: # Catches other potential OS errors during path resolution
             raise ValueError(f"OS error resolving path '{v}': {e}") from e
        except Exception as e: # Catch any other unexpected errors
            raise ValueError(f"Unexpected error processing path '{v}': {e}") from e

    @model_validator(mode='after')
    def validate_paths_and_target_condition(self) -> 'DownloadRequestParameters':
        """
        Validates that:
        1. download_to_path (absolute) exists, is a directory, and is writable.
        2. download_to_path / uuid either does not exist, OR exists as an empty, writable directory.
        """
        base_path_str = self.download_to_path # Already absolute string from field validator
        request_uuid = self.uuid

        try:
            base_path = pathlib.Path(base_path_str)

            # === 1. Validate base_path (must exist and be a writable directory) ===
            if not base_path.exists():
                raise ValueError(f"Base download path '{base_path_str}' does not exist.")
            if not base_path.is_dir():
                raise ValueError(f"Base download path '{base_path_str}' is not a directory.")
            if not os.access(base_path, os.W_OK | os.X_OK):
                 raise PermissionError(f"Insufficient permissions for base path '{base_path_str}'. Cannot write or create subdirectories.")

            # === 2. Construct and validate the target directory path (base_path / uuid) ===
            target_dir = base_path / request_uuid

            if target_dir.exists():
                # --- Target exists: Must be an empty, writable directory ---
                if not target_dir.is_dir():
                    raise ValueError(f"Target path '{target_dir}' exists but is not a directory.")

                # Check if the directory is empty efficiently
                try:
                    # Use a simple loop that breaks on first item found
                    dir_is_empty = True
                    for _ in target_dir.iterdir():
                         dir_is_empty = False
                         break # Found something, no need to check further

                    if not dir_is_empty:
                        raise ValueError(f"Target directory '{target_dir}' exists but is not empty.")

                except PermissionError as e:
                    # Specific error if we can't even list the contents
                    raise PermissionError(f"Cannot check contents of existing target directory '{target_dir}': {e}") from e
                except OSError as e:
                    # Other errors listing contents
                    raise OSError(f"Error checking if existing target directory '{target_dir}' is empty: {e}") from e

                # Check write permissions on the existing empty directory
                if not os.access(target_dir, os.W_OK | os.X_OK):
                    raise PermissionError(f"Target directory '{target_dir}' exists and is empty, but required permissions (write/execute) are missing.")

                # print(f"Debug: Target '{target_dir}' exists and is an empty, writable directory. OK.")

            else:
                # --- Target does not exist: This is acceptable ---
                # We already confirmed the parent (base_path) is writable, so creation should be possible later.
                # print(f"Debug: Target '{target_dir}' does not exist. OK.")
                pass

            # If all checks passed without raising an error
            return self

        except (ValueError, PermissionError, OSError) as e:
             # Re-raise specific validation errors caught above
             raise e
        except Exception as e:
            # Catch any other unexpected errors during path validation
            raise RuntimeError(f"Unexpected error during path validation: {e}") from e

    # Helper property to get the target path easily after validation
    @property
    def target_directory_path(self) -> pathlib.Path:
        """
        Returns the full target directory Path object (base_path / uuid).
        Call this *after* the model has been successfully validated.
        """
        # This assumes validation passed, so paths are valid.
        return pathlib.Path(self.download_to_path) / self.uuid
