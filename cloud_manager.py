import logging
import os

import cloudinary
import cloudinary.uploader
import requests

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloudManager")


class CloudManager:
    def __init__(self):
        if settings.CLOUDINARY_CLOUD_NAME:
            cloudinary.config(
                cloud_name=settings.CLOUDINARY_CLOUD_NAME,
                api_key=settings.CLOUDINARY_API_KEY,
                api_secret=settings.CLOUDINARY_API_SECRET,
                secure=True,
            )

    def upload_model(self):
        if not os.path.exists(settings.MODEL_FILE):
            return
        try:
            cloudinary.uploader.upload(
                settings.MODEL_FILE,
                resource_type="raw",
                public_id=settings.CLOUD_MODEL_NAME,
                overwrite=True,
                invalidate=True,
            )
            logger.info("☁️ Model Uploaded to Cloud.")
        except Exception as e:
            logger.error(f"Upload Failed: {e}")

    def download_model(self):
        try:
            url = cloudinary.utils.cloudinary_url(
                settings.CLOUD_MODEL_NAME, resource_type="raw"
            )[0]
            r = requests.get(url)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(settings.MODEL_FILE), exist_ok=True)
                with open(settings.MODEL_FILE, "wb") as f:
                    f.write(r.content)
                logger.info("☁️ Model Downloaded from Cloud.")
                return True
        except:
            pass
        return False


cloud_manager = CloudManager()
