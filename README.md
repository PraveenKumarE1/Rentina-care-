# DR Screening System

A MATLAB R2023b+ screening scaffold with a Python runner for retinal-image experiments. It provides a runnable, interpretable baseline: image quality control, enhancement, classical vessel/optic-disc/lesion proposals, International Clinical Scale 0-4 grading, lesion overlays, reports, de-identification, SQLite persistence, benchmark metrics, and queue/cost simulation.

## Run

1. Open `DR_Screening_System` in MATLAB.
2. Run `startup`.
3. Run `main`, or call `result = main_pipeline("path/to/fundus.jpg")`.
4. Run `unit_tests` and `integration_tests` from the `tests` folder.

### Python runner

The Python entry point accepts a single fundus image, a DICOM file, or a folder of images:

```powershell
python python_dr_runner.py path\to\fundus.jpg
python python_dr_runner.py path\to\dataset --batch
python python_dr_runner.py --demo
```

For the browser workflow, start `web_ui/server.py` and open `http://127.0.0.1:5000`. Upload a genuine fundus image from one of the datasets below. A normal laptop webcam photograph is not a retinal acquisition and will normally fail the quality gate.

### Browser workflow

1. Open `http://127.0.0.1:5000`.
2. Sign in with an email address. This is a local development login, not production identity verification.
3. Choose English, Hindi, Tamil, Telugu, Malayalam, or Kannada. The selected language changes the main controls and the voice language.
4. Use `Listen` to have the current case review read aloud. Select `Voice control` and allow microphone access to say commands such as `help`, `read result`, `guide me`, `use camera`, `capture photo`, `run screening`, `home`, `about`, or `stop`. Voice commands require the latest Chrome or Edge and work on localhost or HTTPS.
5. Upload a fundus image or use the camera, then choose `Run screening`.
6. Review the result and Recent uploads. History is stored locally in `database/upload_history.sqlite3`.
7. Open `Your screening history` and press `Refresh guide` to get a personal summary based on past uploads.

The personal guide learns only safe summaries from past screening history. Only separately labeled images can be used for supervised model training; past screening uploads must not be treated as training labels automatically.

The Home and About sections explain the screening safeguards and model limitations. Each completed screening also presents a model-review panel with the active method, confidence, image-quality feedback, and an explicit clinician-review reminder. Low-confidence predictions are flagged for additional review; this is a safety feature, not a clinical-performance claim.

### Train a retinal model

The repository includes `train_retinal_model.py`, a small PyTorch classifier for the APTOS 2019 format. After downloading the labeled APTOS files, arrange them as:

```text
datasets/aptos2019/
	train.csv
	train_images/
		<id_code>.png
```

The CSV must contain `id_code` and `diagnosis` columns. Train and save a checkpoint with:

```powershell
python train_retinal_model.py --data-dir datasets\aptos2019 --epochs 10 --output models\aptos_retinal_cnn.pt
```

To download the Kaggle competition files with `kagglehub`, first authenticate with your own Kaggle account in the project virtual environment. Do not put an API token in source code:

```powershell
.venv\Scripts\Activate.ps1
python -c "import kagglehub; kagglehub.login()"
python -c "import kagglehub; print(kagglehub.competition_download('aptos2019-blindness-detection'))"
```

The account must have access to the APTOS competition and accept its rules if Kaggle requests it. The current environment reached Kaggle but returned `UnauthenticatedError` because no Kaggle account was signed in.

The script reports validation accuracy, balanced accuracy, and quadratic Cohen kappa. When `models/user_retinal_cnn.pt` exists, the browser runner automatically uses that trained PyTorch CNN for gradable images; otherwise it transparently uses the interpretable development baseline.

### Retinal-image diabetes-risk research model

Diabetic-retinopathy grades are **not** diabetes labels. To enable the separate retinal-image diabetes-risk estimate, train on a dataset in which every retinal image has a verified diabetes status (from A1C or plasma-glucose criteria). Use a CSV with `id_code` and `diabetes`, where `diabetes` is `0` for no diabetes and `1` for diabetes:

```powershell
python train_retinal_model.py --data-dir datasets\diabetes_retina --target-column diabetes --task diabetes_risk --epochs 10 --output models\diabetes_retinal_risk_cnn.pt
```

The browser will then show “Higher” or “Lower retinal-image diabetes risk” for gradable images. This is a research screening estimate only—not a diagnosis—and must be confirmed with A1C or plasma-glucose testing and clinical review.

## Retinal datasets

The project can process the image files from these datasets through the Python runner:

| Dataset | Primary use | Source |
| --- | --- | --- |
| APTOS 2019 Blindness Detection | Diabetic retinopathy severity classification, grades 0-4 | [Kaggle competition](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| IDRiD | DR grading plus lesion and optic-disc annotations | [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| DRIVE | Retinal vessel extraction and segmentation | [Grand Challenge](https://drive.grand-challenge.org/) |
| Messidor-2 | External diabetic retinopathy evaluation and validation | [ADCIS](https://www.adcis.net/en/third-party/messidor2/) |

Place downloaded image files in separate folders so each dataset can be evaluated independently:

```text
datasets/
	aptos2019/train_images/
	idrid/images/
	drive/images/
	messidor2/images/
```

Run a batch and generate CSV/JSON output for any image folder:

```powershell
python python_dr_runner.py datasets\aptos2019\train_images --batch
python python_dr_runner.py datasets\idrid\images --batch
python python_dr_runner.py datasets\drive\images --batch
python python_dr_runner.py datasets\messidor2\images --batch
```

Each batch creates `batch_reports\batch_results.csv` and `batch_reports\batch_results.json` inside the selected dataset folder. Patient/image identifiers are derived from filenames when no separate metadata file is supplied.

The current Python grader is a classical rule-based development baseline. It does not load APTOS labels or train a neural network, and it does not use DRIVE vessel masks or IDRiD lesion annotations as ground truth. Dataset labels and annotations must be connected to a trained model and an evaluation script before reporting sensitivity, specificity, AUROC, or clinical performance.

Required runtime capabilities for the baseline are Image Processing Toolbox and (for DICOM/database features) Medical Imaging Toolbox and Database Toolbox. The code does not claim clinical performance targets until trained weights and external validation data are supplied.

## Structure

`backend/pipeline` contains quality, segmentation, and grading. `backend/explainability` contains evidence adapters. `frontend/report_generator.m` creates HTML and PDF when `exportgraphics` is available. `database` contains SQLite persistence and schema. `simulink` contains analytic KPI and cost functions plus the binary model specification. `validation` contains benchmark metrics.

## Clinical and deployment boundary

The included classical detector is a development baseline, not a medical device. It must not be used for diagnosis or patient management. Add ophthalmologist-reviewed labels, trained ResNet/Inception/DenseNet/ViT and vessel U-Net assets, calibration, Grad-CAM target layers, dataset-specific validation, cybersecurity review, and regulatory approval before clinical use. A general REST listener is represented by `api_server` and should be hosted through MATLAB Production Server or a separately secured gateway.

## Sample call

```matlab
startup
result = main_pipeline(imread('sample_fundus.jpg'));
report = report_generator(result,'DR-0001','reports','en');
disp(result.summary)
```
