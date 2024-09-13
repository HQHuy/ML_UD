from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from model import db, UploadedFile
from flask_migrate import Migrate
import os

app = Flask(__name__)
app.secret_key = '2002'

# Thiết lập đường dẫn tới file database
basedir = os.path.abspath(os.path.dirname(__file__))
database_path = os.path.join(basedir, 'db', 'dataweb.db')

# Cấu hình SQLAlchemy để kết nối tới SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{database_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'upload')  
db.init_app(app)
migrate = Migrate(app, db)

# Tạo thư mục uploads nếu chưa tồn tại
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET'])
def home():
    uploaded_files = UploadedFile.query.order_by(UploadedFile.upload_time.desc()).all()
    return render_template('home/home.html', uploaded_files=uploaded_files)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)
            new_file = UploadedFile(filename=filename)
            db.session.add(new_file)
            db.session.commit()
            flash(f'File "{filename}" đã được tải lên thành công!', 'info')
            return redirect(url_for('upload_file'))
        else:
            flash('Không có file nào được chọn hoặc file không hợp lệ.', 'danger')
            return redirect(url_for('upload_file'))
    return render_template('uploads/uploads.html')

@app.route('/delete-file/<filename>', methods=['POST'])
def delete_file(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            file_to_delete = UploadedFile.query.filter_by(filename=filename).first()
            if file_to_delete:
                db.session.delete(file_to_delete)
                db.session.commit()
            flash(f'Tệp tin "{filename}" đã được xóa thành công.', 'info')
        else:
            flash(f'Tệp tin "{filename}" không tồn tại.', 'warning')
    except Exception as e:
        flash(f'Có lỗi xảy ra: {str(e)}', 'danger')

    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
