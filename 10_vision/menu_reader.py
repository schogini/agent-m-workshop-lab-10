import sys
import os
from pathlib import Path
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from agno.agent import Agent
from agno.media import Image
from common import make_model

from dotenv import load_dotenv

from agno.models.openai import OpenAIResponses

load_dotenv(ROOT / ".env")

VISION_MODEL_ID = os.getenv(
    "VISION_MODEL_ID",
    "inclusionai/ling-3.0-flash-vl:free"
)
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
)


class Menu(BaseModel):
    breakfast: list[str] = Field(default_factory=list)
    lunch: list[str] = Field(default_factory=list)
    dinner: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


agent = Agent(
    # model=make_model(),
    model=OpenAIResponses(id=VISION_MODEL_ID),
    output_schema=Menu,
    instructions=[
        "Read the menu image carefully.",
        "Extract only food visibly listed in the image.",
        "Do not invent missing meals.",
    ],
)

if __name__ == "__main__":
    default_image = Path(__file__).parent / "menu.png"
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_image

    if not image_path.exists():
        print(f"Menu image not found: {image_path}")
        print("Take a photo of a menu and run:")
        print("  uv run python 10_vision/menu_reader.py path/to/menu.jpg")
        raise SystemExit(1)

    result = agent.run(
        "Extract this menu into breakfast, lunch and dinner.",
        images=[Image(filepath=str(image_path))],
    )

    if isinstance(result.content, Menu):
        print(result.content.model_dump_json(indent=2))
    else:
        print(result.content)
