from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.data_processing import process_data
from app.model_selection import select_model
from app.model_training import train_model
from app.model_evaluation import evaluate_model
from app.prediction import predict
from app.monitoring import monitor_model_performance
from app.logging_config import logger
from app.model_updater import start_model_updater
import joblib
import json
import threading

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")

    # Process the uploaded file
    data, profile_report = await process_data(file)

    # Select the best model
    model, problem_type = select_model(data)

    # Train the model
    trained_model, X_test, y_test, cv_scores = train_model(model, data, problem_type)

    # Evaluate the model
    evaluation_results = evaluate_model(trained_model, (X_test, y_test), problem_type)

    # Save the model
    joblib.dump(trained_model, 'static/trained_model.joblib')

    # Schedule background task for model monitoring
    background_tasks.add_task(monitor_model_performance, trained_model, X_test, y_test, problem_type)

    logger.info("Model training and evaluation completed successfully")

    return {
        "message": "File processed successfully",
        "profile_report": profile_report,
        "model_name": type(trained_model).__name__,
        "cv_scores": cv_scores.tolist(),
        **evaluation_results
    }

@app.get("/download_model")
async def download_model():
    return FileResponse('static/trained_model.joblib', filename='trained_model.joblib')

@app.post("/predict")
async def make_prediction(input_data: dict):
    try:
        prediction = predict(input_data)
        logger.info(f"Prediction made: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

@app.get("/model_info")
async def get_model_info():
    try:
        model = joblib.load('static/trained_model.joblib')
        model_info = {
            "model_type": type(model).__name__,
            "feature_importance": model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
            "parameters": model.get_params()
        }
        return model_info
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        return {"error": str(e)}

# if __name__ == "__main__":
#     # Start model updater in a separate thread
#     updater_thread = threading.Thread(target=start_model_updater)
#     updater_thread.start()

#     uvicorn.run(app, host="0.0.0.0", port=8000)