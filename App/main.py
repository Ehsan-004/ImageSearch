import io
import uuid
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Request
from fastapi import UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from Inference.model import get_model
from Inference.search import search
from Inference.chroma_configs import get_pr_collection, get_client

BASE_DIRECTORY = Path("/home/phoenix/ehsan/projects/datasets/GLAMI/images")


model = get_model()
chroma_client = get_client()
product_collection = get_pr_collection(chroma_client)
version = "v1"

app = FastAPI(
    version=version
)

app.mount(
    "/static",
    StaticFiles(directory="App/static/"),
    name="static"
)

app.mount("/data", StaticFiles(directory="/home/phoenix/ehsan/projects/datasets/GLAMI/images/"), name="data")


templates = Jinja2Templates(directory="App/templates/")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Image matching engine"
        }
    )


@app.post("/search", response_class=HTMLResponse)
async def search_image(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Form(10)
):
    import uuid, io
    from PIL import Image
    from fastapi.templating import Jinja2Templates

    # read image in memory
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # save query image (for display)
    job_id = str(uuid.uuid4())
    query_path = f"inputs/{job_id}_query.png"
    full_query_path = f"App/static/{query_path}"
    img.save(full_query_path)

    D, I = search(img, top_k)


    matched_ids = [str(idx) for idx in I[0]]

    results = product_collection.get(ids=matched_ids)

    output_results = []

    for dist, meta in zip(D[0], results["metadatas"]):
        path = meta["path"]
        output_results.append({
            "path": path.replace("/home/phoenix/ehsan/projects/datasets/GLAMI/images/", "data/"),
            "distance": float(dist),
            "category": str(meta.get("category", "Unknown"))
        })

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "query_image": f"/static/{query_path}",
            "results": output_results,
            "top_k": top_k
        }
    )

    # # search (Stage1 + Stage2)
    # results = search(img, top_k)  # returns list of tuples: [(category, distance, idx), ...]

    # # prepare output list
    # output_results = []
    # for cat, dist, idx in results:
    #     # fetch metadata from chroma collection
    #     res = product_collection.get(ids=[str(idx)])
    #     if not res or not res.get("metadatas"):
    #         continue
    #     meta = res["metadatas"][0]

    #     path = meta.get("path")
    #     if not path:
    #         continue

    #     output_results.append({
    #         "path": path.replace("/home/phoenix/ehsan/projects/ImageSearch/data", "/data"),
    #         "distance": float(dist),
    #         "category": str(meta.get("category", "Unknown"))
    #     })

    # return templates.TemplateResponse(
    #     "results.html",
    #     {
    #         "request": request,
    #         "query_image": f"/static/{query_path}",
    #         "results": output_results,
    #         "top_k": top_k
    #     }
    # )
