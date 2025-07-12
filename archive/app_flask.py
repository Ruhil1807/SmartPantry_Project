from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
from bson.objectid import ObjectId

from scripts.db import (
    get_items, get_alerts, insert_item,
    get_user_by_email, insert_user,
    get_item_by_id, update_item, delete_item_by_id
)
from scripts.predict import predict_spoilage

app = Flask(__name__)
app.secret_key = "your-secret-key"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    raw_items = get_items()
    items, restock_alerts = [], []

    for item in raw_items:
        try:
            expiry_raw = str(item.get("expiry", "")).strip()
            expiry_date = datetime.fromisoformat(expiry_raw) if "T" in expiry_raw or " " in expiry_raw else datetime.strptime(expiry_raw, "%Y-%m-%d")
            item["days_left"] = (expiry_date - datetime.today()).days
        except:
            item["days_left"] = "N/A"

        try:
            if item.get("restock_threshold") and item["quantity"] < int(item["restock_threshold"]):
                restock_alerts.append(f"⚠️ {item['item']} is below restock threshold!")
        except:
            pass

        items.append(item)

    alerts = get_alerts()
    return render_template("index.html", items=items, alerts=alerts, restock_alerts=restock_alerts, predict_spoilage=predict_spoilage)

@app.route('/add', methods=['GET', 'POST'])
def add_item():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        item = request.form['item']
        category = request.form['category']
        added_on = request.form['added_on']
        expiry = request.form['expiry']
        restock_threshold = request.form.get('restock_threshold')
        barcode = request.form.get('barcode')
        quantity = int(request.form.get("quantity", 1))
        image_url = None

        if not all([item, category, added_on, expiry]):
            return render_template('add_item.html', error="All required fields must be filled.")

        if expiry <= added_on:
            return render_template('add_item.html', error="Expiry must be after the added date.")

        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = f"{UPLOAD_FOLDER}/{filename}"

        new_item = {
            "item": item,
            "category": category,
            "added_on": added_on,
            "expiry": expiry,
            "quantity": quantity
        }
        if restock_threshold:
            new_item["restock_threshold"] = int(restock_threshold)
        if barcode:
            new_item["barcode"] = barcode
        if image_url:
            new_item["image"] = image_url

        insert_item(new_item)
        return redirect(url_for('index'))

    return render_template('add_item.html')

@app.route('/edit/<item_id>', methods=['GET', 'POST'])
def edit_item(item_id):
    if 'user_email' not in session:
        return redirect(url_for('login'))

    item = get_item_by_id(item_id)
    if not item:
        return "Item not found", 404

    if request.method == 'POST':
        updated = {
            "item": request.form['item'],
            "category": request.form['category'],
            "added_on": request.form['added_on'],
            "expiry": request.form['expiry'],
            "restock_threshold": int(request.form.get('restock_threshold', 0)),
            "quantity": int(request.form.get('quantity', 1)),
            "barcode": request.form.get('barcode')
        }
        update_item(item_id, updated)
        return redirect(url_for('index'))

    return render_template('edit_item.html', item=item)

@app.route('/delete/<item_id>', methods=['POST'])
def delete_item(item_id):
    if 'user_email' not in session:
        return redirect(url_for('login'))

    delete_item_by_id(item_id)
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if get_user_by_email(email):
            return render_template("signup.html", error="Email already registered.")
        insert_user({
            "email": email,
            "password": generate_password_hash(password)
        })
        return redirect(url_for('login'))
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = get_user_by_email(email)
        if user and check_password_hash(user["password"], password):
            session['user_email'] = email
            return redirect(url_for('index'))
        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
