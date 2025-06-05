import os
import gzip
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="CGN Corpus Inspector", description="Browse and inspect CGN corpus files"
)

# Get base path from environment or use default
CGN_BASE_PATH = os.getenv("CGN", "../data/CGN_2.0.3")

# Set up templates
templates = Jinja2Templates(directory="templates")


class AudioFileInfo(BaseModel):
    filename: str
    component: str
    language: str
    full_path: str
    has_annotation: bool


class AnnotationWord(BaseModel):
    ref: str
    start_time: float
    end_time: float
    word: str


class AnnotationTurn(BaseModel):
    ref: str
    speaker: str
    start_time: float
    end_time: float
    words: List[AnnotationWord]


def get_components() -> List[str]:
    """Get list of available components (comp-a to comp-o)"""
    audio_path = Path(CGN_BASE_PATH) / "data" / "audio" / "wav"
    if not audio_path.exists():
        return []

    components = []
    for item in audio_path.iterdir():
        if item.is_dir() and item.name.startswith("comp-"):
            components.append(item.name)

    return sorted(components)


def get_languages_for_component(component: str) -> List[str]:
    """Get available languages (nl/vl) for a component"""
    component_path = Path(CGN_BASE_PATH) / "data" / "audio" / "wav" / component
    if not component_path.exists():
        return []

    languages = []
    for item in component_path.iterdir():
        if item.is_dir() and item.name in ["nl", "vl"]:
            languages.append(item.name)

    return sorted(languages)


def get_audio_files(component: str, language: str) -> List[AudioFileInfo]:
    """Get list of audio files for a component and language"""
    audio_path = Path(CGN_BASE_PATH) / "data" / "audio" / "wav" / component / language
    annot_path = (
        Path(CGN_BASE_PATH)
        / "data"
        / "annot"
        / "xml"
        / "skp-ort"
        / component
        / language
    )

    if not audio_path.exists():
        return []

    files = []
    for audio_file in audio_path.glob("*.wav"):
        # Check if corresponding annotation exists
        annotation_file = annot_path / f"{audio_file.stem}.skp.gz"
        has_annotation = annotation_file.exists()

        files.append(
            AudioFileInfo(
                filename=audio_file.name,
                component=component,
                language=language,
                full_path=str(audio_file),
                has_annotation=has_annotation,
            )
        )

    return sorted(files, key=lambda x: x.filename)


def parse_annotation_file(
    component: str, language: str, filename: str
) -> List[AnnotationTurn]:
    """Parse the annotation XML file for a given audio file"""
    base_name = filename.replace(".wav", "")
    annot_path = (
        Path(CGN_BASE_PATH)
        / "data"
        / "annot"
        / "xml"
        / "skp-ort"
        / component
        / language
        / f"{base_name}.skp.gz"
    )

    if not annot_path.exists():
        raise HTTPException(status_code=404, detail="Annotation file not found")

    # Extract and parse the gzipped XML
    with gzip.open(annot_path, "rt", encoding="utf-8") as f:
        xml_content = f.read()

    # Parse XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing XML: {str(e)}")

    turns = []
    for tau in root.findall("tau"):
        words = []
        for tw in tau.findall("tw"):
            word = AnnotationWord(
                ref=tw.get("ref", ""),
                start_time=float(tw.get("tb", 0)),
                end_time=float(tw.get("te", 0)),
                word=tw.get("w", ""),
            )
            words.append(word)

        turn = AnnotationTurn(
            ref=tau.get("ref", ""),
            speaker=tau.get("s", ""),
            start_time=float(tau.get("tb", 0)),
            end_time=float(tau.get("te", 0)),
            words=words,
        )
        turns.append(turn)

    return turns


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page showing available components"""
    components = get_components()
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "components": components, "cgn_path": CGN_BASE_PATH},
    )


@app.get("/component/{component}", response_class=HTMLResponse)
async def component_view(request: Request, component: str):
    """View languages available for a component"""
    if component not in get_components():
        raise HTTPException(status_code=404, detail="Component not found")

    languages = get_languages_for_component(component)
    return templates.TemplateResponse(
        "component.html",
        {"request": request, "component": component, "languages": languages},
    )


@app.get("/component/{component}/{language}", response_class=HTMLResponse)
async def language_view(request: Request, component: str, language: str):
    """View audio files for a component and language"""
    if component not in get_components():
        raise HTTPException(status_code=404, detail="Component not found")

    if language not in get_languages_for_component(component):
        raise HTTPException(
            status_code=404, detail="Language not found for this component"
        )

    audio_files = get_audio_files(component, language)
    return templates.TemplateResponse(
        "files.html",
        {
            "request": request,
            "component": component,
            "language": language,
            "audio_files": audio_files,
        },
    )


@app.get("/annotation/{component}/{language}/{filename}")
async def get_annotation(component: str, language: str, filename: str):
    """Get annotation data for an audio file"""
    turns = parse_annotation_file(component, language, filename)
    return {
        "component": component,
        "language": language,
        "filename": filename,
        "turns": turns,
    }


@app.get("/view/{component}/{language}/{filename}", response_class=HTMLResponse)
async def view_annotation(
    request: Request, component: str, language: str, filename: str
):
    """View annotation in a nice HTML format"""
    try:
        turns = parse_annotation_file(component, language, filename)
    except HTTPException as e:
        if e.status_code == 404:
            # No annotation file found, but still show audio player
            turns = []
        else:
            raise e

    return templates.TemplateResponse(
        "annotation.html",
        {
            "request": request,
            "component": component,
            "language": language,
            "filename": filename,
            "turns": turns,
        },
    )


@app.get("/audio/{component}/{language}/{filename}")
async def serve_audio(component: str, language: str, filename: str):
    """Serve audio files"""
    audio_path = (
        Path(CGN_BASE_PATH) / "data" / "audio" / "wav" / component / language / filename
    )

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(path=str(audio_path), media_type="audio/wav", filename=filename)


@app.get("/api/components")
async def api_components():
    """API endpoint to get all components"""
    return {"components": get_components()}


@app.get("/api/component/{component}/languages")
async def api_languages(component: str):
    """API endpoint to get languages for a component"""
    return {"languages": get_languages_for_component(component)}


@app.get("/api/component/{component}/{language}/files")
async def api_files(component: str, language: str):
    """API endpoint to get files for a component and language"""
    return {"files": get_audio_files(component, language)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
