from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from nst_deploy import run_nst

app = FastAPI(title="Neural Style Transfer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Serve the /style folder and the HTML file ──────────────────────
app.mount("/style", StaticFiles(directory="style"), name="style")

@app.get("/")
def serve_ui():
    return FileResponse("nst_ui.html")

# ── Health check ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Main endpoint ──────────────────────────────────────────────────
@app.post("/style-transfer")
async def style_transfer(
    content:       UploadFile = File(...),
    style:         UploadFile = File(...),
    learning_rate: float      = Form(0.05),
    epochs:        int        = Form(50),
    alpha:         float      = Form(30.0),
    beta:          float      = Form(10.0),
):
    if not content.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Content file must be an image.")
    if not style.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Style file must be an image.")
    if not (0.01 <= learning_rate <= 0.2):
        raise HTTPException(status_code=400, detail="learning_rate must be between 0.0001 and 0.1.")
    if not (50 <= epochs <= 200):
        raise HTTPException(status_code=400, detail="epochs must be between 100 and 2000.")

    content_bytes = await content.read()
    style_bytes   = await style.read()

    try:
        result_bytes = run_nst(
            content_bytes = content_bytes,
            style_bytes   = style_bytes,
            learning_rate = learning_rate,
            epochs        = epochs,
            alpha         = alpha,
            beta          = beta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NST failed: {str(e)}")

    return Response(content=result_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)