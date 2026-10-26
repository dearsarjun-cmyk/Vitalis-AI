# VITALIS AI — public-ready website

A responsive chatbot website for the VITALIS language model. The model is a small educational Transformer initialized from random weights; it is **not** a pretrained model, a validated medical system, or comparable to a large commercial assistant.

## Run locally

Requirements: Python 3.11 and `uv`.

```bash
uv sync
uv run python hello.py train --steps 2000
uv run python web_app.py
```

Open `http://127.0.0.1:8000`. In VS Code, choose **VITALIS Website** in **Run and Debug** and press the green Run button. Training is a separate step and should be done first. The tiny built-in sample corpus only tests the training pipeline; it does not teach meaningful language ability.

Put UTF-8 `.txt` files you are allowed to use under `data/corpus/` before training. Never include private, personal, confidential, or copyrighted text without permission in a public deployment.

## Publish it to the public internet

This project is **prepared for deployment but is not published yet**. A public URL requires a hosting account and deployment from your own account; this package cannot create an account or publish to a hosting provider on your behalf.

### Recommended route: GitHub + Render

1. Create a new repository on GitHub and upload the contents of this folder.
2. Train VITALIS on your own machine first:
   ```bash
   uv run python hello.py train --steps 2000
   ```
3. For actual model responses on the public site, the deployment needs the matching `vitalis_model.pt` checkpoint and `tokenizer.json` created by training. Those files are intentionally excluded by `.gitignore` by default because model files can be large. If you choose to publish your trained model, remove the `vitalis_model.pt` and `vitalis_latest.pt` ignore lines from `.gitignore`, then add `vitalis_model.pt` and `tokenizer.json` to the repository. Do **not** upload training documents or private data. If the checkpoint is too large for GitHub, use a hosting service's persistent disk or model-artifact storage instead.
4. In Render, create a **New Web Service** and connect your GitHub repository. Render can use `render.yaml`; alternatively set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python web_app.py`
   - **Health check path:** `/health`
5. Choose an instance with enough memory for PyTorch and your model. Very small/free instances may run out of memory or be too slow. Wait for deployment to finish; Render will show a public `https://…onrender.com` URL.
6. Open that URL and test it. The site can load without a checkpoint, but chat requests will say the model needs to be trained/deployed until `vitalis_model.pt` and `tokenizer.json` are present.

## Important before making it public

- Anyone who has the URL can access the website and send prompts to your model. Do not enter sensitive information.
- This is a prototype HTTP server, not a hardened production service. Before sharing widely, add authentication, stronger abuse/rate limits, request monitoring, and HTTPS through the hosting provider. Keep server dependencies updated.
- Responses can be incoherent or wrong. Do not use VITALIS for diagnosis, treatment, or other medical decisions.
- A tiny corpus and 2,000 CPU training steps are not enough to create a capable general-purpose chatbot.

## VS Code Run configurations

- **VITALIS Website** — starts the web server.
- **VITALIS Train (2000 steps)** — trains the model; do this separately.
- **VITALIS Chat (terminal)** — terminal chat.

The web server binds to `0.0.0.0` and uses the platform `PORT` environment variable so it can run on a host. Locally, open `http://127.0.0.1:8000`.
