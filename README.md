# ML CI/CD: Drug Classification Demo

This project demonstrates model training, evaluation, ONNX export, GitHub Actions CI/CD, CML reporting, and deployment of a static browser app to a Hugging Face Space.

> **Safety:** This is an educational demonstration only. It is not a medical tool and must not be used to select or prescribe medication.

## Start here (Windows / VS Code)

### A. Train the model

1. Extract this ZIP.
2. In VS Code, choose **File → Open Folder** and open the extracted `ml-cicd-gradio-lite` folder.
3. Open **Terminal → New Terminal**.
4. Copy and run these commands one at a time (do not copy the `PS ...>` prompt):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_dataset.py
python train.py
```

If PowerShell blocks activation, run this once in the terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

After training, the project creates `model/drug_model.joblib`, `model/drug_model.onnx`, and evaluation files in `results/`.

### B. Open the web app

1. Keep the terminal in the main project folder.
2. Run:

```powershell
python -m http.server 8000
```

3. Open a browser and visit **http://localhost:8000/app/**.
4. Wait for the status to say **Model loaded**. Enter sample values and click **Predict class**.
5. To stop the local server, return to the terminal and press `Ctrl+C`.

The web page checks both the deployed model path and the local project model path automatically. You do **not** need to copy the model file into another folder.

## Project structure

- `.github/workflows/ci.yml` — GitHub Actions workflow
- `download_dataset.py` — downloads the sample dataset
- `train.py` — trains/evaluates a Random Forest model and exports ONNX
- `data/` — dataset downloaded at setup
- `model/` — generated model files
- `results/` — metrics, summary JSON, and confusion matrix
- `app/index.html` — browser app using ONNX Runtime Web

## GitHub and Hugging Face deployment

Once local training and browser testing work:

1. Create a GitHub repository and push this project.
2. In GitHub repository **Settings → Secrets and variables → Actions**, add `HF_TOKEN` (a Hugging Face token with write access) and `HF_SPACE_ID` (`YOUR_USERNAME/drug-classification-lite`).
3. Create a Hugging Face Space with **Static** SDK and the name `drug-classification-lite`.
4. Push to the `main` branch and inspect the repository's **Actions** tab.

The workflow is a learning template. Check the Actions logs if any deployment step fails; repository permissions and secret values must be configured correctly.

## Dataset and metrics

The sample dataset contains 200 rows. The model's accuracy can vary if the dataset or training configuration changes. Always report metrics from your own run rather than treating an example result as guaranteed.
