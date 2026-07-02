# Optional Firestore logger. Requires google-cloud-firestore and credentials.
import os
try:
    from google.cloud import firestore  # type: ignore
except Exception:
    firestore = None

class FirestoreLogger:
    def __init__(self, collection_name: str = "mediscan_predictions"):
        if firestore is None:
            raise RuntimeError("google-cloud-firestore is not installed.")
        self.client = firestore.Client()
        self.collection = self.client.collection(collection_name)

    def log_prediction(self, ref_id: str, image_name: str, label: str, confidence: float):
        doc = {
            "ref_id": ref_id,
            "image_name": image_name,
            "label": label,
            "confidence": float(confidence),
            "timestamp": firestore.SERVER_TIMESTAMP,
        }
        self.collection.document(ref_id).set(doc, merge=True)
