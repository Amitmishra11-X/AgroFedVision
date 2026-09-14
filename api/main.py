"""
AgroFedVision FastAPI Backend

React
  ↓
FastAPI
  ↓
ml_model.predict2
  ↓
Image + Sensor + UAV
  ↓
Decision Fusion + Explainable AI
  ↓
JSON
"""

import os
import sys
import uuid
import shutil
import zipfile
import tempfile
import traceback
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ML_MODEL_DIR = BASE_DIR / "ml_model"
UPLOAD_DIR = BASE_DIR / "api_uploads"
IMAGE_UPLOAD_DIR = UPLOAD_DIR / "images"
UAV_UPLOAD_DIR = UPLOAD_DIR / "uav"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UAV_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# PYTHON PATH
# ==========================================================

if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))


# ==========================================================
# EXISTING ML PIPELINE
# ==========================================================

try:
    from predict2 import predict_and_recommend
except Exception as e:
    raise RuntimeError(
        "Could not import ml_model.predict2.\n"
        f"Error: {e}"
    )


# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="AgroFedVision API",
    description=(
        "Multimodal crop health prediction API using "
        "leaf images, sensor data and UAV multispectral information."
    ),
    version="1.2.0",
)


# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# STATIC ACCESS
# ==========================================================

app.mount(
    "/results",
    StaticFiles(directory=str(RESULTS_DIR)),
    name="results",
)

app.mount(
    "/uploads",
    StaticFiles(directory=str(UPLOAD_DIR)),
    name="uploads",
)


# ==========================================================
# OPENAPI FILE SCHEMA FIX
# ==========================================================

# ==========================================================
# OPENAPI FILE SCHEMA FIX
# ==========================================================

# Swagger UI needs array items to be declared as:
#   type: string
#   format: binary
# for multiple-file upload controls.
#
# FastAPI handles List[UploadFile] correctly at runtime, but depending on
# the FastAPI/Pydantic version, the generated OpenAPI may expose the items
# only as "string". This custom OpenAPI function fixes the documentation
# schema without changing the prediction pipeline.

from typing import Any, Dict


def custom_openapi() -> Dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    multipart_schemas = [
        "/predict/image",
        "/predict/multimodal",
    ]

    for route_path in multipart_schemas:
        path_item = schema.get("paths", {}).get(route_path, {})
        post_operation = path_item.get("post", {})
        request_body = post_operation.get("requestBody", {})
        content = request_body.get("content", {})
        multipart = content.get("multipart/form-data", {})
        body_schema = multipart.get("schema", {})

        # Resolve FastAPI's generated request-body schema.
        if "$ref" in body_schema:
            ref_name = body_schema["$ref"].split("/")[-1]
            body_schema = schema.get("components", {}).get(
                "schemas", {}
            ).get(ref_name, {})

        properties = body_schema.get("properties", {})
        image_schema = properties.get("image")

        if image_schema:
            # Preserve the array, but explicitly make every item a file.
            image_schema["type"] = "array"
            image_schema["items"] = {
                "type": "string",
                "format": "binary",
            }
            image_schema["description"] = (
                "Upload one or more leaf images."
            )

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


# ==========================================================
# BASIC ROUTES
# ==========================================================

@app.get("/")
def root():
    return {
        "project": "AgroFedVision",
        "status": "running",
        "version": "1.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "AgroFedVision FastAPI",
    }


@app.get("/crops")
def get_crops():
    try:
        from crop_registry import list_available_crops

        return {
            "crops": list_available_crops()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ==========================================================
# FILE HELPERS
# ==========================================================

async def save_upload(
    upload_file: UploadFile,
    destination_dir: Path,
) -> Path:

    original_name = Path(
        upload_file.filename or "upload"
    ).name

    extension = Path(
        original_name
    ).suffix.lower()

    unique_name = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    destination = destination_dir / unique_name

    with destination.open("wb") as buffer:
        shutil.copyfileobj(
            upload_file.file,
            buffer,
        )

    return destination


def validate_image_upload(upload_file: UploadFile) -> None:
    """Validate an uploaded leaf image file."""
    filename = Path(upload_file.filename or "").name
    extension = Path(filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    if extension not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format: {extension or 'unknown'}. "
                "Use JPG, JPEG, PNG, WEBP or BMP."
            ),
        )


def safe_extract_zip(
    zip_path: Path,
    extract_dir: Path,
) -> Path:
    """
    Safely extract a UAV ZIP and return the directory that
    actually contains the multispectral TIFF capture files.

    Supports both:

        ZIP/
            IMG_0000_1.tif
            IMG_0000_3.tif
            IMG_0000_4.tif
            ...

    and:

        ZIP/
            000/
                IMG_0000_1.tif
                IMG_0000_3.tif
                IMG_0000_4.tif
                ...
    """

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # ZIP security check
    # ------------------------------------------------------

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as z:

        root = extract_dir.resolve()

        for member in z.infolist():

            member_path = (
                extract_dir / member.filename
            ).resolve()

            if not str(member_path).startswith(
                str(root) + os.sep
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Unsafe path detected inside UAV ZIP."
                    ),
                )

        z.extractall(extract_dir)

    # ------------------------------------------------------
    # Find directories containing TIFF files
    # ------------------------------------------------------

    def tif_files(directory: Path):
        return [
            p for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in (
                ".tif",
                ".tiff",
            )
        ]

    # First check extraction root.
    root_tifs = tif_files(extract_dir)

    if root_tifs:
        capture_dir = extract_dir

    else:
        # Search recursively for the first directory
        # containing TIFF files.
        candidates = []

        for p in extract_dir.rglob("*"):
            if p.is_dir():
                files = tif_files(p)
                if files:
                    candidates.append(
                        (p, files)
                    )

        if not candidates:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The UAV ZIP was extracted, but no "
                    "TIFF/TIF files were found."
                ),
            )

        # Prefer a directory containing *_1.tif files,
        # because those are the capture anchors expected
        # by batch_extract().
        capture_dir = None

        for directory, files in candidates:
            if any(
                f.name.lower().endswith("_1.tif")
                or f.name.lower().endswith("_1.tiff")
                for f in files
            ):
                capture_dir = directory
                break

        if capture_dir is None:
            capture_dir = candidates[0][0]

    # ------------------------------------------------------
    # Validate complete capture anchors
    # ------------------------------------------------------

    capture_files = [
        p for p in capture_dir.iterdir()
        if p.is_file()
        and (
            p.name.lower().endswith("_1.tif")
            or p.name.lower().endswith("_1.tiff")
        )
    ]

    if not capture_files:
        raise HTTPException(
            status_code=400,
            detail=(
                "No *_1.tif multispectral capture files "
                "were found. The UAV ZIP must contain "
                "complete multispectral captures."
            ),
        )

    print(
        f"[UAV API] Capture directory: {capture_dir}"
    )

    print(
        f"[UAV API] Capture count: "
        f"{len(capture_files)}"
    )

    return capture_dir


def validate_uav_zip(upload_file: UploadFile):
    filename = (
        upload_file.filename or ""
    ).lower()

    if not filename.endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail=(
                "UAV upload must be a ZIP file containing "
                "the multispectral TIFF/TIF captures."
            ),
        )


# ==========================================================
# SENSOR INPUT
# ==========================================================

def build_sensor_reading(
    N: Optional[float],
    P: Optional[float],
    K: Optional[float],
    temperature: Optional[float],
    humidity: Optional[float],
    ph: Optional[float],
    rainfall: Optional[float],
    soil_moisture: Optional[float],
    soil_type: Optional[float],
    sunlight_exposure: Optional[float],
    wind_speed: Optional[float],
    co2_concentration: Optional[float],
    organic_matter: Optional[float],
    irrigation_frequency: Optional[float],
    crop_density: Optional[float],
    pest_pressure: Optional[float],
    fertilizer_usage: Optional[float],
    growth_stage: Optional[float],
    urban_area_proximity: Optional[float],
    water_source_type: Optional[float],
    frost_risk: Optional[float],
    water_usage_efficiency: Optional[float],
):

    sensor_reading = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "soil_type": soil_type,
        "sunlight_exposure": sunlight_exposure,
        "wind_speed": wind_speed,
        "co2_concentration": co2_concentration,
        "organic_matter": organic_matter,
        "irrigation_frequency": irrigation_frequency,
        "crop_density": crop_density,
        "pest_pressure": pest_pressure,
        "fertilizer_usage": fertilizer_usage,
        "growth_stage": growth_stage,
        "urban_area_proximity": urban_area_proximity,
        "water_source_type": water_source_type,
        "frost_risk": frost_risk,
        "water_usage_efficiency": water_usage_efficiency,
    }

    if all(
        value is None
        for value in sensor_reading.values()
    ):
        return None

    return sensor_reading


# ==========================================================
# IMAGE ONLY
# ==========================================================

@app.post("/predict/image")
async def predict_image_api(
    crop: str = Form(...),
    image: List[UploadFile] = File(
        ...,
        description="Upload one or more leaf images."
    ),
):

    if not image:
        raise HTTPException(
            status_code=400,
            detail="At least one leaf image is required."
        )

    image_paths = []
    for uploaded_image in image:
        validate_image_upload(uploaded_image)
        saved_path = await save_upload(
            uploaded_image,
            IMAGE_UPLOAD_DIR,
        )
        image_paths.append(str(saved_path))

    try:

        report = predict_and_recommend(
            crop=crop,
            image_paths=image_paths,
            uav_capture_folder=None,
            uav_image_paths=None,
            uav_image_id=None,
            sensor_reading=None,
        )

        # Replace internal filesystem paths with API paths
        report = sanitize_response_paths(
            report
        )

        return report

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ==========================================================
# MULTIMODAL
# ==========================================================

@app.post("/predict/multimodal")
async def predict_multimodal_api(

    crop: str = Form(...),

    image: List[UploadFile] = File(
        ...,
        description="Upload one or more leaf images."
    ),

    uav_zip: Optional[
        UploadFile
    ] = File(
        default=None,
        description=(
            "Optional ZIP containing UAV "
            "multispectral TIFF/TIF captures."
        ),
    ),

    N: Optional[float] = Form(None),
    P: Optional[float] = Form(None),
    K: Optional[float] = Form(None),

    temperature: Optional[float] = Form(None),
    humidity: Optional[float] = Form(None),
    ph: Optional[float] = Form(None),
    rainfall: Optional[float] = Form(None),

    soil_moisture: Optional[float] = Form(None),
    soil_type: Optional[float] = Form(None),
    sunlight_exposure: Optional[float] = Form(None),
    wind_speed: Optional[float] = Form(None),
    co2_concentration: Optional[float] = Form(None),
    organic_matter: Optional[float] = Form(None),
    irrigation_frequency: Optional[float] = Form(None),
    crop_density: Optional[float] = Form(None),
    pest_pressure: Optional[float] = Form(None),
    fertilizer_usage: Optional[float] = Form(None),
    growth_stage: Optional[float] = Form(None),
    urban_area_proximity: Optional[float] = Form(None),
    water_source_type: Optional[float] = Form(None),
    frost_risk: Optional[float] = Form(None),
    water_usage_efficiency: Optional[float] = Form(None),
):

    # ------------------------------------------------------
    # SAVE LEAF IMAGE
    # ------------------------------------------------------

    if not image:
        raise HTTPException(
            status_code=400,
            detail="At least one leaf image is required."
        )

    image_paths = []
    for uploaded_image in image:
        validate_image_upload(uploaded_image)
        saved_path = await save_upload(
            uploaded_image,
            IMAGE_UPLOAD_DIR,
        )
        image_paths.append(str(saved_path))

    # ------------------------------------------------------
    # SENSOR
    # ------------------------------------------------------

    sensor_reading = build_sensor_reading(
        N=N,
        P=P,
        K=K,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall,
        soil_moisture=soil_moisture,
        soil_type=soil_type,
        sunlight_exposure=sunlight_exposure,
        wind_speed=wind_speed,
        co2_concentration=co2_concentration,
        organic_matter=organic_matter,
        irrigation_frequency=irrigation_frequency,
        crop_density=crop_density,
        pest_pressure=pest_pressure,
        fertilizer_usage=fertilizer_usage,
        growth_stage=growth_stage,
        urban_area_proximity=urban_area_proximity,
        water_source_type=water_source_type,
        frost_risk=frost_risk,
        water_usage_efficiency=water_usage_efficiency,
    )

    # ------------------------------------------------------
    # UAV ZIP
    # ------------------------------------------------------

    temp_dir = None
    uav_capture_folder = None

    if uav_zip is not None:

        validate_uav_zip(
            uav_zip
        )

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="agrofedvision_uav_"
            )
        )

        zip_path = (
            temp_dir / "uav_upload.zip"
        )

        with zip_path.open("wb") as buffer:
            shutil.copyfileobj(
                uav_zip.file,
                buffer,
            )

        uav_capture_folder = safe_extract_zip(
            zip_path,
            temp_dir / "extracted",
        )

    # ------------------------------------------------------
    # RUN EXISTING PIPELINE
    # ------------------------------------------------------

    try:

        report = predict_and_recommend(
            crop=crop,
            image_paths=image_paths,
            uav_capture_folder=(
                str(uav_capture_folder)
                if uav_capture_folder
                else None
            ),
            uav_image_paths=None,
            uav_image_id=None,
            sensor_reading=sensor_reading,
        )

        report = sanitize_response_paths(
            report
        )

        return report

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        if temp_dir is not None:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


# ==========================================================
# SENSOR ONLY
# ==========================================================

@app.post("/analyze/sensor")
def analyze_sensor(

    N: Optional[float] = None,
    P: Optional[float] = None,
    K: Optional[float] = None,

    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    ph: Optional[float] = None,
    rainfall: Optional[float] = None,

    soil_moisture: Optional[float] = None,
    soil_type: Optional[float] = None,
    sunlight_exposure: Optional[float] = None,
    wind_speed: Optional[float] = None,
    co2_concentration: Optional[float] = None,
    organic_matter: Optional[float] = None,
    irrigation_frequency: Optional[float] = None,
    crop_density: Optional[float] = None,
    pest_pressure: Optional[float] = None,
    fertilizer_usage: Optional[float] = None,
    growth_stage: Optional[float] = None,
    urban_area_proximity: Optional[float] = None,
    water_source_type: Optional[float] = None,
    frost_risk: Optional[float] = None,
    water_usage_efficiency: Optional[float] = None,
):

    sensor_reading = build_sensor_reading(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall,
        soil_moisture,
        soil_type,
        sunlight_exposure,
        wind_speed,
        co2_concentration,
        organic_matter,
        irrigation_frequency,
        crop_density,
        pest_pressure,
        fertilizer_usage,
        growth_stage,
        urban_area_proximity,
        water_source_type,
        frost_risk,
        water_usage_efficiency,
    )

    if sensor_reading is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "At least one sensor value "
                "must be supplied."
            ),
        )

    try:

        from sensor_utils import (
            predict_sensor,
            load_sensor_artifacts,
        )

        artifacts = (
            load_sensor_artifacts()
        )

        if artifacts is None:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Sensor model artifacts "
                    "not found."
                ),
            )

        return predict_sensor(
            sensor_reading,
            artifacts=artifacts,
        )

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ==========================================================
# UAV ONLY
# ==========================================================

@app.post("/analyze/uav")
async def analyze_uav(

    uav_zip: UploadFile = File(
        ...,
        description=(
            "ZIP containing complete UAV "
            "multispectral TIFF/TIF captures."
        ),
    ),

):

    validate_uav_zip(
        uav_zip
    )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="agrofedvision_uav_only_"
        )
    )

    try:

        zip_path = (
            temp_dir / "uav_upload.zip"
        )

        with zip_path.open("wb") as buffer:
            shutil.copyfileobj(
                uav_zip.file,
                buffer,
            )

        capture_dir = safe_extract_zip(
            zip_path,
            temp_dir / "extracted",
        )

        from predict2 import (
            predict_uav_capture
        )

        result = predict_uav_capture(
            capture_folder=str(
                capture_dir
            ),
            uav_paths=None,
        )

        return result

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


# ==========================================================
# RESPONSE PATH SANITIZATION
# ==========================================================

def sanitize_response_paths(obj):
    """
    Convert internal Windows/Linux filesystem paths
    into API-relative paths where possible.

    Example:
        C:\\...\\api_uploads\\images\\abc.jpg
    becomes:
        /uploads/images/abc.jpg
    """

    if isinstance(obj, dict):

        return {
            key: sanitize_response_paths(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):

        return [
            sanitize_response_paths(value)
            for value in obj
        ]

    if isinstance(obj, str):

        normalized = obj.replace(
            "\\",
            "/",
        )

        upload_marker = "/api_uploads/"

        if upload_marker in normalized:

            relative = normalized.split(
                upload_marker,
                1,
            )[1]

            return "/uploads/" + relative

        results_marker = "/results/"

        if results_marker in normalized:

            relative = normalized.split(
                results_marker,
                1,
            )[1]

            return "/results/" + relative

        return obj

    return obj


# ==========================================================
# SHUTDOWN
# ==========================================================

@app.on_event("shutdown")
def shutdown_event():

    print(
        "AgroFedVision API shutting down."
    )
