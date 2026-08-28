COLORS = {
    "idle": "#2e9d55", "scanning": "#2583d8", "paused": "#e0a21b", "error": "#d43b3b",
    "stopped": "#777777", "starting": "#2583d8", "stopping": "#777777",
}


def make_icon(state="idle", size=32):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    color = COLORS.get(state, COLORS["stopped"])
    draw.ellipse((2, 2, size - 3, size - 3), fill=color, outline="white", width=max(1, size // 16))
    draw.rectangle((size * 9 // 32, size * 8 // 32, size * 13 // 32, size * 24 // 32), fill="white")
    draw.rectangle((size * 18 // 32, size * 8 // 32, size * 22 // 32, size * 24 // 32), fill="white")
    if state == "scanning":
        draw.polygon(((size * 11 // 32, size * 8 // 32), (size * 24 // 32, size // 2),
                      (size * 11 // 32, size * 24 // 32)), fill="white")
    return image
