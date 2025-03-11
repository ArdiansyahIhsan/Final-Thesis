from flask import Flask, request, render_template, send_file, session, redirect, url_for
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from fpdf import FPDF
import matplotlib.pyplot as plt
import uuid
import os
import io
import tempfile

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = tempfile.gettempdir()
# Lokasi untuk menyimpan gambar grafik
UPLOAD__FOLDER = os.path.join('static', 'uploads')

# Cek apakah folder 'static/uploads' sudah ada, jika belum buat foldernya
if not os.path.exists(UPLOAD__FOLDER):
    os.makedirs(UPLOAD__FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('dataset')
        if not file:
            return "No file uploaded"
        
        # Save dataset file in a temporary directory
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Save file path in session
        session['dataset_path'] = file_path
        
        # Read dataset
        dataset = pd.read_csv(file_path)
        table_html = dataset.to_html(classes='table table-bordered centered-table', header=True, index=False)
        table_html = table_html.replace('<thead>', '<thead class="header-row">')
        return render_template('dataset.html', data=table_html, columns=dataset.columns)

    return render_template('index2.html')

@app.route('/batasan', methods=['GET'])
def batasan():
    return render_template('batasan.html')

@app.route('/visualisasi_page')
def visualisasi_page():
    # Baca dataset
    df = pd.read_csv(session.get('dataset_path'))
    df['date'] = pd.to_datetime(df['date'])

    # Kelompokkan data berdasarkan interval 10 hari dan hitung total quantity
    df_grouped = df.resample('10D', on='date').sum()

    # Plot grafik
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped.index, df_grouped['menu__quantity'], marker='o')
    plt.xlabel('Date')
    plt.ylabel('Total Quantity')
    plt.grid(True)
    plt.xticks(df_grouped.index, rotation=90)

    # Simpan grafik ke file sementara
    img_filename = f'{uuid.uuid4()}.png'
    img_path = os.path.join(UPLOAD__FOLDER, img_filename)
    plt.savefig(img_path)
    
    # Simpan path gambar dalam session
    session['img_path'] = img_filename  

    # Tampilkan halaman visualisasi
    return redirect(url_for('visualisasi'))

@app.route('/visualisasi')
def visualisasi():
    img_path = session.get('img_path')
    if not img_path:
        return redirect(url_for('index'))
    
    # Kirim path gambar ke template
    return render_template('visualisasi.html', img_path=session['img_path'])

@app.route('/show_dataset', methods=['GET'])
def show_dataset():
    if 'dataset_path' not in session:
        return redirect(url_for('index'))

    file_path = session['dataset_path']
    dataset = pd.read_csv(file_path)
    table_html = dataset.to_html(classes='table table-bordered centered-table', header=True, index=False)
    table_html = table_html.replace('<thead>', '<thead class="header-row">')

    return render_template('dataset.html', data=table_html, columns=dataset.columns)


@app.route('/process', methods=['POST'])
def process():
    support = float(request.form.get('support')) / 100
    confidence = float(request.form.get('confidence')) / 100

    if 'dataset_path' not in session:
        return redirect(url_for('index'))

    file_path = session['dataset_path']
    dataset = pd.read_csv(file_path)

    selected_features = ['date', 'menu__name', 'menu__quantity']
    dataset = dataset[selected_features]

    dataset = dataset.dropna()

    dataset = dataset.pivot_table(index=['date'], columns='menu__name', values='menu__quantity').fillna(0)
    dataset = dataset.astype("int32")

    dataset = dataset.applymap(lambda x: 1 if x > 0 else 0).astype(int)

    frequent_itemsets = fpgrowth(dataset, min_support=support, use_colnames=True)
    frequent_itemsets['support_count'] = frequent_itemsets['support'] * len(dataset)
    frequent_itemsets = frequent_itemsets.sort_values(by='support_count', ascending=False)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['Aturan Asosiasi'] = 'Jika memesan ' + rules['antecedents'] + ', maka cenderung memesan ' + rules['consequents']
    
    rules = rules[rules.apply(lambda x: len(x['antecedents'].split(', ')) + len(x['consequents'].split(', ')) <= 4, axis=1)]

    rules = rules[['Aturan Asosiasi', 'support', 'confidence', 'lift']]
    rules_html = rules.to_html(classes='table table-striped centered-table', header=True, index=False)
    rules_html = rules_html.replace('<thead>', '<thead class="header-row">')

    rules_csv = rules.to_csv(index=False)
    rules_path = os.path.join(UPLOAD_FOLDER, 'rules.csv')
    with open(rules_path, 'w') as f:
        f.write(rules_csv)
    session['rules_path'] = rules_path

    return render_template('results.html', tables=[rules_html], download_link=True)

@app.route('/download')
def download():
    if 'rules_path' not in session:
        return redirect(url_for('index'))

    rules_path = session['rules_path']
    rules = pd.read_csv(rules_path)
    
    buffer = io.BytesIO()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    col_width_rule = pdf.w / 2
    col_width = pdf.w / 8
    row_height = pdf.font_size
    columns = ['Aturan Asosiasi', 'Support', 'Confidence', 'Lift']
    
    # Header
    for col in columns:
        if col == 'Aturan Asosiasi':
            pdf.cell(col_width_rule, row_height * 2, col, border=1, align='L')
        else:
            pdf.cell(col_width, row_height * 2, col, border=1, align='C')
    pdf.ln(row_height * 2)

    # Data
    for i in range(len(rules)):
        # Aturan Asosiasi menggunakan multi_cell untuk menghindari teks keluar dari kolom
        pdf.multi_cell(col_width_rule, row_height, rules['Aturan Asosiasi'].iloc[i], border=1, align='L')
        x = pdf.get_x()
        y = pdf.get_y()
        
        # Kembalikan posisi X dan Y untuk kolom berikutnya di baris yang sama
        pdf.set_xy(x + col_width_rule, y - row_height)

        pdf.cell(col_width, row_height, f"{rules['support'].iloc[i]:.4f}", border=1, align='C')
        pdf.cell(col_width, row_height, f"{rules['confidence'].iloc[i]:.4f}", border=1, align='C')
        pdf.cell(col_width, row_height, f"{rules['lift'].iloc[i]:.4f}", border=1, align='C')
        
        pdf.ln(row_height)

    pdf_output = pdf.output(dest='S').encode('latin1')
    buffer.write(pdf_output)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name='rules.pdf', mimetype='application/pdf')


if __name__ == '__main__':
    app.run(debug=True)
