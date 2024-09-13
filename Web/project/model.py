from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class UploadedFile(db.Model):
    __tablename__ = 'uploaded_files'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=False)
    upload_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __init__(self, filename):
        self.filename = filename
        self.upload_time = datetime.utcnow()

    def __repr__(self):
        return f'<UploadedFile {self.filename} - {self.upload_time}>'
