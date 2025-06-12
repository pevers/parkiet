from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from pathlib import Path
import uvicorn

app = FastAPI(title="Prompts Viewer")

# Mount static files for serving audio files
app.mount("/audio", StaticFiles(directory="../data/prompts"), name="audio")

PROMPTS_DIR = Path("../data/prompts")


def get_emotion_emoji(emotion: str) -> str:
    """Convert emotion string to emoji"""
    emotion_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "surprised": "😲",
        "fearful": "😨",
        "disgusted": "🤢",
        "neutral": "😐",
    }
    return emotion_map.get(emotion.lower(), "😐")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    with open("src/templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/folders")
async def get_folders():
    """Get list of all folders in the prompts directory"""
    folders = []
    for item in PROMPTS_DIR.iterdir():
        if item.is_dir():
            # Check for subfolders
            subfolders = []
            for subitem in item.iterdir():
                if subitem.is_dir():
                    has_prompts = (subitem / "prompts.json").exists()
                    subfolders.append(
                        {
                            "name": subitem.name,
                            "path": str(subitem.relative_to(PROMPTS_DIR)),
                            "has_prompts": has_prompts,
                        }
                    )

            folders.append({"name": item.name, "subfolders": subfolders})

    return {"folders": folders}


@app.get("/api/prompts/{folder_path:path}")
async def get_prompts(folder_path: str):
    """Get prompts for a specific folder"""
    prompts_file = PROMPTS_DIR / folder_path / "prompts.json"

    if not prompts_file.exists():
        raise HTTPException(status_code=404, detail="Prompts file not found")

    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)

        # Process prompts to add emojis and format data
        processed_prompts = []
        total_duration = 0

        for prompt in prompts_data:
            processed_prompt = {"type": prompt["type"], "data": prompt["data"].copy()}

            # Add emotion emojis
            if prompt["type"] == "single":
                processed_prompt["data"]["emotion_emoji"] = get_emotion_emoji(
                    prompt["data"]["emotion"]
                )
                total_duration += prompt["data"]["duration"]
            elif prompt["type"] == "pair":
                processed_prompt["data"]["emotion_a_emoji"] = get_emotion_emoji(
                    prompt["data"]["emotion_a"]
                )
                processed_prompt["data"]["emotion_b_emoji"] = get_emotion_emoji(
                    prompt["data"]["emotion_b"]
                )
                total_duration += prompt["data"]["duration"]

            # Format duration for display
            duration_seconds = (
                prompt["data"]["duration"] / 1000
            )  # Convert from ms to seconds
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            processed_prompt["data"]["duration_formatted"] = f"{minutes}:{seconds:02d}"

            processed_prompts.append(processed_prompt)

        # Format total duration
        total_seconds = total_duration / 1000
        total_minutes = int(total_seconds // 60)
        total_secs = int(total_seconds % 60)
        total_hours = int(total_minutes // 60)
        total_mins = total_minutes % 60

        if total_hours > 0:
            total_duration_formatted = (
                f"{total_hours}:{total_mins:02d}:{total_secs:02d}"
            )
        else:
            total_duration_formatted = f"{total_mins}:{total_secs:02d}"

        return {
            "prompts": processed_prompts,
            "total_duration": total_duration_formatted,
            "folder_path": folder_path,
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in prompts file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading prompts: {str(e)}")


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs("src/templates", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
